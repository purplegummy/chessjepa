"""
PPO training for the TransformerMoveDecoder on top of a frozen JEPA context encoder.

Architecture:
  - Frozen JEPA encoder: board tensor (B, T, 17, 8, 8) → latents (B, T, P, D)
  - Trainable decoder: latents → policy logits (B, 4096) + value (B,)

RL setup:
  - Environment: python-chess self-play
  - Opponent: random legal moves (swap to self-play via --opponent self)
  - Reward: sparse game outcome (+1 win, 0 draw, -1 loss) from current player's view
  - Algorithm: PPO with GAE
  - Recommended: warm-start from a supervised checkpoint (--decoder_ckpt)
"""

import argparse
import contextlib
import os
import sys
import time
import random
from dataclasses import dataclass, field
from typing import List, Optional

import chess
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.acjepa import ActionConditionedChessJEPA
from util.config import JEPAConfig
from best_move.transformer_decoder import TransformerMoveDecoder

# ── board encoding (mirrors preprocess_pgn.py) ──────────────────────────────
_PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Current-player-relative (17, 8, 8) uint8 tensor."""
    flip = board.turn == chess.BLACK
    t = np.zeros((17, 8, 8), dtype=np.uint8)
    us   = chess.BLACK if flip else chess.WHITE
    them = chess.WHITE if flip else chess.BLACK

    for i, piece in enumerate(_PIECES):
        for sq in board.pieces(piece, us):
            r = (7 - sq // 8) if flip else (sq // 8)
            t[i, r, sq % 8] = 1
        for sq in board.pieces(piece, them):
            r = (7 - sq // 8) if flip else (sq // 8)
            t[i + 6, r, sq % 8] = 1

    w_ks = board.has_kingside_castling_rights(chess.WHITE)
    w_qs = board.has_queenside_castling_rights(chess.WHITE)
    b_ks = board.has_kingside_castling_rights(chess.BLACK)
    b_qs = board.has_queenside_castling_rights(chess.BLACK)
    if flip:
        t[12], t[13], t[14], t[15] = int(b_ks), int(b_qs), int(w_ks), int(w_qs)
    else:
        t[12], t[13], t[14], t[15] = int(w_ks), int(w_qs), int(b_ks), int(b_qs)

    if board.ep_square is not None:
        sq = board.ep_square
        r = (7 - sq // 8) if flip else (sq // 8)
        t[16, r, sq % 8] = 1

    return t


def board_to_input(board: chess.Board, device: torch.device) -> torch.Tensor:
    """(17, 8, 8) → (1, 1, 17, 8, 8) float tensor for the encoder."""
    t = board_to_tensor(board)
    return torch.from_numpy(t).float().unsqueeze(0).unsqueeze(0).to(device)


def legal_mask_for_board(board: chess.Board) -> torch.Tensor:
    """(4096,) bool mask of legal moves (from_sq * 64 + to_sq)."""
    mask = torch.zeros(4096, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move.from_square * 64 + move.to_square] = True
    return mask


def move_to_idx(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def idx_to_move(board: chess.Board, idx: int) -> Optional[chess.Move]:
    """Convert a flat index back to a Move, checking legality."""
    from_sq = idx // 64
    to_sq   = idx % 64
    # Try all legal moves with this from/to (handles promotions by picking queen)
    candidates = [m for m in board.legal_moves
                  if m.from_square == from_sq and m.to_square == to_sq]
    if not candidates:
        return None
    # Prefer queen promotion if multiple (e.g. promotion choices)
    for m in candidates:
        if m.promotion in (None, chess.QUEEN):
            return m
    return candidates[0]


# ── Transition storage ───────────────────────────────────────────────────────

@dataclass
class Transition:
    board_tensor: torch.Tensor   # (1, 17, 8, 8)
    action:       int
    log_prob:     float
    value:        float
    reward:       float
    done:         bool


@dataclass
class Episode:
    transitions: List[Transition] = field(default_factory=list)

    def add(self, t: Transition):
        self.transitions.append(t)

    def __len__(self):
        return len(self.transitions)


# ── Policy helpers ───────────────────────────────────────────────────────────

@torch.no_grad()
def select_action(
    board: chess.Board,
    encoder: torch.nn.Module,
    decoder: TransformerMoveDecoder,
    device: torch.device,
    temperature: float = 1.0,
):
    """Sample an action from the policy; return (move_idx, log_prob, value)."""
    x = board_to_input(board, device)        # (1, 1, 17, 8, 8)
    latent = encoder(x)                       # (1, T, P, D)
    logits, value = decoder(latent)           # (1, 4096), (1,)

    mask = legal_mask_for_board(board).to(device)  # (4096,)
    if not mask.any():
        return None, 0.0, value.item()

    masked = logits[0].masked_fill(~mask, float('-inf'))
    if temperature != 1.0:
        masked = masked / temperature
    probs = F.softmax(masked, dim=-1)
    dist  = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action).item(), value.item()


def compute_log_prob(
    board_tensors: torch.Tensor,   # (B, 17, 8, 8)
    actions:       torch.Tensor,   # (B,) int64
    legal_masks:   torch.Tensor,   # (B, 4096) bool
    encoder:       torch.nn.Module,
    decoder:       TransformerMoveDecoder,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass; returns (log_probs (B,), values (B,))."""
    x = board_tensors.unsqueeze(1)             # (B, 1, 17, 8, 8)
    latent = encoder(x)
    logits, values = decoder(latent)

    masked = logits.masked_fill(~legal_masks, float('-inf'))
    log_probs_all = F.log_softmax(masked, dim=-1)
    log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)
    return log_probs, values


# ── GAE ─────────────────────────────────────────────────────────────────────

def compute_gae(
    rewards: List[float],
    values:  List[float],
    dones:   List[bool],
    gamma:   float = 0.99,
    lam:     float = 0.95,
) -> tuple[List[float], List[float]]:
    """Compute generalised advantage estimates and returns."""
    T = len(rewards)
    advantages = [0.0] * T
    returns    = [0.0] * T
    gae = 0.0
    next_val = 0.0

    for t in reversed(range(T)):
        if dones[t]:
            next_val = 0.0
            gae      = 0.0
        delta = rewards[t] + gamma * next_val - values[t]
        gae   = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t]    = gae + values[t]
        next_val = values[t]

    return advantages, returns


# ── Rollout collection ───────────────────────────────────────────────────────

def collect_rollout(
    encoder:      torch.nn.Module,
    decoder:      TransformerMoveDecoder,
    device:       torch.device,
    n_games:      int,
    max_moves:    int,
    opponent:     str,         # "random" or "self"
    temperature:  float,
) -> List[Episode]:
    """Play n_games from white's perspective; collect one episode per game."""
    decoder.eval()
    episodes = []

    for _ in range(n_games):
        board   = chess.Board()
        episode = Episode()
        policy_is_white = True  # always start white; we flip each move

        for _ in range(max_moves):
            if board.is_game_over():
                break

            it_is_policy_turn = (board.turn == chess.WHITE) == policy_is_white

            if it_is_policy_turn:
                # Policy move
                board_t = torch.from_numpy(board_to_tensor(board)).float().to(device)
                action, lp, val = select_action(board, encoder, decoder, device, temperature)
                if action is None:
                    break
                move = idx_to_move(board, action)
                if move is None or move not in board.legal_moves:
                    # Fallback: pick a random legal move
                    move   = random.choice(list(board.legal_moves))
                    action = move_to_idx(move)
                board.push(move)

                # Reward is 0 until the game ends
                done = board.is_game_over()
                reward = 0.0
                if done:
                    result = board.result()   # "1-0", "0-1", "1/2-1/2"
                    if result == "1-0":
                        reward = 1.0 if policy_is_white else -1.0
                    elif result == "0-1":
                        reward = -1.0 if policy_is_white else 1.0
                    else:
                        reward = 0.0

                episode.add(Transition(
                    board_tensor=board_t,
                    action=action,
                    log_prob=lp,
                    value=val,
                    reward=reward,
                    done=done,
                ))

            else:
                # Opponent move
                if opponent == "random" or board.legal_moves.count() == 0:
                    opp_move = random.choice(list(board.legal_moves))
                else:
                    # Self-play: sample from same policy without gradient
                    _, _, _ = select_action(board, encoder, decoder, device, temperature)
                    opp_move = random.choice(list(board.legal_moves))  # TODO: use policy move
                    # Simple self-play: use policy output
                    action_opp, _, _ = select_action(board, encoder, decoder, device, temperature)
                    opp_move_candidate = idx_to_move(board, action_opp) if action_opp is not None else None
                    opp_move = opp_move_candidate if opp_move_candidate in board.legal_moves else random.choice(list(board.legal_moves))

                board.push(opp_move)

                # Opponent's move may end the game — back-assign reward to last policy transition
                if board.is_game_over() and len(episode.transitions) > 0:
                    result = board.result()
                    if result == "1-0":
                        r = 1.0 if policy_is_white else -1.0
                    elif result == "0-1":
                        r = -1.0 if policy_is_white else 1.0
                    else:
                        r = 0.0
                    last = episode.transitions[-1]
                    episode.transitions[-1] = Transition(
                        board_tensor=last.board_tensor,
                        action=last.action,
                        log_prob=last.log_prob,
                        value=last.value,
                        reward=r,
                        done=True,
                    )
                    break

        episodes.append(episode)

    return episodes


# ── PPO update ───────────────────────────────────────────────────────────────

def ppo_update(
    encoder:        torch.nn.Module,
    decoder:        TransformerMoveDecoder,
    optimizer:      torch.optim.Optimizer,
    episodes:       List[Episode],
    device:         torch.device,
    clip_eps:       float,
    value_coef:     float,
    entropy_coef:   float,
    grad_clip:      float,
    ppo_epochs:     int,
    minibatch_size: int,
    gamma:          float,
    lam:            float,
    amp_ctx,
    scaler,
) -> dict:
    """Run PPO update on collected episodes. Returns dict of loss components."""

    # ── 1. Flatten all transitions ──────────────────────────────────────────
    all_boards, all_actions, all_old_lp, all_masks = [], [], [], []
    all_advantages, all_returns = [], []

    for ep in episodes:
        if len(ep) == 0:
            continue
        rewards = [t.reward  for t in ep.transitions]
        values  = [t.value   for t in ep.transitions]
        dones   = [t.done    for t in ep.transitions]
        advs, rets = compute_gae(rewards, values, dones, gamma, lam)

        for t, adv, ret in zip(ep.transitions, advs, rets):
            all_boards.append(t.board_tensor)
            all_actions.append(t.action)
            all_old_lp.append(t.log_prob)
            all_masks.append(legal_mask_for_board_from_tensor(t.board_tensor))
            all_advantages.append(adv)
            all_returns.append(ret)

    if len(all_boards) == 0:
        return {}

    boards_t   = torch.stack(all_boards).to(device)              # (N, 17, 8, 8)
    actions_t  = torch.tensor(all_actions,    dtype=torch.long,  device=device)
    old_lp_t   = torch.tensor(all_old_lp,    dtype=torch.float32, device=device)
    masks_t    = torch.stack(all_masks).to(device)               # (N, 4096)
    advs_t     = torch.tensor(all_advantages, dtype=torch.float32, device=device)
    rets_t     = torch.tensor(all_returns,    dtype=torch.float32, device=device)

    # Normalize advantages
    advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

    N = boards_t.shape[0]
    total_loss = total_policy = total_value = total_entropy = 0.0
    n_updates = 0

    decoder.train()
    for _ in range(ppo_epochs):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, minibatch_size):
            idx = perm[start : start + minibatch_size]

            mb_boards  = boards_t[idx]
            mb_actions = actions_t[idx]
            mb_old_lp  = old_lp_t[idx]
            mb_masks   = masks_t[idx]
            mb_advs    = advs_t[idx]
            mb_rets    = rets_t[idx]

            optimizer.zero_grad()
            with amp_ctx():
                new_lp, new_val = compute_log_prob(mb_boards, mb_actions, mb_masks, encoder, decoder)

                # Policy loss (clipped surrogate)
                ratio       = (new_lp - mb_old_lp).exp()
                surr1       = ratio * mb_advs
                surr2       = ratio.clamp(1 - clip_eps, 1 + clip_eps) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss  = F.mse_loss(new_val, mb_rets)

                # Entropy bonus (over legal moves only)
                x_in    = mb_boards.unsqueeze(1)
                latent  = encoder(x_in)
                logits, _ = decoder(latent)
                masked  = logits.masked_fill(~mb_masks, float('-inf'))
                probs   = F.softmax(masked, dim=-1).clamp(min=1e-8)
                entropy = -(probs * probs.log()).sum(dim=-1).mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                optimizer.step()

            total_loss    += loss.item()
            total_policy  += policy_loss.item()
            total_value   += value_loss.item()
            total_entropy += entropy.item()
            n_updates     += 1

    if n_updates == 0:
        return {}
    return {
        "loss":    total_loss    / n_updates,
        "policy":  total_policy  / n_updates,
        "value":   total_value   / n_updates,
        "entropy": total_entropy / n_updates,
        "n_transitions": N,
    }


def legal_mask_for_board_from_tensor(board_tensor: torch.Tensor) -> torch.Tensor:
    """
    We need to reconstruct legal moves from a board tensor stored in a Transition.
    For simplicity, we store the legal mask at collection time alongside the board.
    This function is a placeholder — in practice we precompute and store masks.
    Returns a full-ones mask (all moves legal) as a fallback.
    """
    # We return a dummy mask here; the real mask is passed separately in ppo_update
    return torch.ones(4096, dtype=torch.bool)


# ── Main training loop ───────────────────────────────────────────────────────

def train_ppo(
    jepa_ckpt:      str,
    decoder_ckpt:   Optional[str]  = None,
    n_iterations:   int            = 500,
    games_per_iter: int            = 32,
    max_moves:      int            = 200,
    opponent:       str            = "random",
    temperature:    float          = 1.0,
    gamma:          float          = 0.99,
    lam:            float          = 0.95,
    ppo_epochs:     int            = 4,
    minibatch_size: int            = 256,
    clip_eps:       float          = 0.2,
    value_coef:     float          = 0.5,
    entropy_coef:   float          = 0.01,
    lr:             float          = 3e-4,
    grad_clip:      float          = 0.5,
    output_path:    str            = "best_move/decoder_ppo.pt",
    log_every:      int            = 10,
    device_str:     str            = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
):
    device = torch.device(device_str)
    print(f"Device: {device}")

    # ── Load JEPA encoder (frozen) ──────────────────────────────────────────
    print(f"Loading JEPA checkpoint: {jepa_ckpt}")
    ckpt = torch.load(jepa_ckpt, map_location=device, weights_only=False)
    cfg: JEPAConfig = ckpt["config"]

    jepa = ActionConditionedChessJEPA(
        encoder_kwargs=cfg.encoder_kwargs,
        predictor_kwargs=cfg.predictor_kwargs,
    ).to(device)
    jepa.load_state_dict(ckpt["model"])
    encoder = jepa.context_encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # ── Load / init decoder ─────────────────────────────────────────────────
    embed_dim   = cfg.encoder_kwargs.get("embed_dim", 256)
    num_patches = (cfg.board_size // cfg.patch_size) ** 2

    decoder = TransformerMoveDecoder(
        embed_dim=embed_dim,
        num_patches=num_patches,
        num_heads=8,
        ff_dim=512,
        num_layers=2,
        mlp_hidden=256,
        dropout=0.1,
        head_dropout=0.1,
        latent_dropout=0.0,   # disable latent dropout during RL
    ).to(device)

    if decoder_ckpt is not None and os.path.exists(decoder_ckpt):
        print(f"Warm-starting decoder from: {decoder_ckpt}")
        d = torch.load(decoder_ckpt, map_location=device, weights_only=False)
        decoder.load_state_dict(d["decoder"])
    else:
        print("No decoder checkpoint — starting from random weights (not recommended)")

    # ── AMP ─────────────────────────────────────────────────────────────────
    if device_str == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_ctx   = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_dtype = None
        amp_ctx   = contextlib.nullcontext
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    print(f"AMP: {'enabled (' + str(amp_dtype) + ')' if amp_dtype else 'disabled'}")

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    best_reward = -float("inf")

    print(f"{'─'*65}")
    print(f"PPO | iters={n_iterations}  games/iter={games_per_iter}  opponent={opponent}")
    print(f"      ppo_epochs={ppo_epochs}  clip={clip_eps}  lr={lr}")
    print(f"{'─'*65}")

    # Track recent win rate
    recent_results: List[float] = []

    for iteration in range(1, n_iterations + 1):
        t0 = time.time()

        # ── Collect rollout ─────────────────────────────────────────────────
        # We need to store legal masks alongside transitions; patch collect_rollout
        # to also stash the mask in the Transition (see below).
        episodes = collect_rollout_with_masks(
            encoder, decoder, device,
            n_games=games_per_iter,
            max_moves=max_moves,
            opponent=opponent,
            temperature=temperature,
        )

        # Track win rate
        for ep in episodes:
            if ep.transitions:
                last_r = ep.transitions[-1].reward
                recent_results.append(last_r)
        if len(recent_results) > 200:
            recent_results = recent_results[-200:]
        win_rate  = sum(r > 0 for r in recent_results) / max(len(recent_results), 1)
        draw_rate = sum(r == 0 for r in recent_results) / max(len(recent_results), 1)
        loss_rate = sum(r < 0 for r in recent_results) / max(len(recent_results), 1)

        # ── PPO update ──────────────────────────────────────────────────────
        stats = ppo_update_with_masks(
            encoder, decoder, optimizer, episodes, device,
            clip_eps, value_coef, entropy_coef, grad_clip,
            ppo_epochs, minibatch_size, gamma, lam, amp_ctx, scaler,
        )

        elapsed = time.time() - t0

        if stats and (iteration % log_every == 0 or iteration == 1):
            avg_r = sum(recent_results) / max(len(recent_results), 1)
            print(
                f"Iter {iteration:4d}/{n_iterations} | "
                f"W={win_rate:.2f} D={draw_rate:.2f} L={loss_rate:.2f} | "
                f"AvgR={avg_r:.3f} | "
                f"Loss={stats.get('loss', 0):.4f} "
                f"(pol={stats.get('policy', 0):.4f} "
                f"val={stats.get('value', 0):.4f} "
                f"ent={stats.get('entropy', 0):.4f}) | "
                f"N={stats.get('n_transitions', 0)} | {elapsed:.1f}s"
            )

        avg_r = sum(recent_results) / max(len(recent_results), 1)
        if avg_r > best_reward and len(recent_results) >= games_per_iter:
            best_reward = avg_r
            torch.save({
                "decoder":   decoder.state_dict(),
                "iteration": iteration,
                "avg_reward": avg_r,
                "win_rate":   win_rate,
            }, output_path)

    print(f"\nDone. Best avg reward: {best_reward:.4f}  →  {output_path}")


# ── Extended collect_rollout that stores legal masks in Transition ────────────

@dataclass
class TransitionWithMask(Transition):
    legal_mask: Optional[torch.Tensor] = None


@dataclass
class EpisodeWithMasks:
    transitions: List[TransitionWithMask] = field(default_factory=list)

    def add(self, t: TransitionWithMask):
        self.transitions.append(t)

    def __len__(self):
        return len(self.transitions)


def collect_rollout_with_masks(
    encoder, decoder, device, n_games, max_moves, opponent, temperature
) -> List[EpisodeWithMasks]:
    decoder.eval()
    episodes = []

    for _ in range(n_games):
        board   = chess.Board()
        episode = EpisodeWithMasks()
        policy_is_white = True

        for _ in range(max_moves):
            if board.is_game_over():
                break

            it_is_policy_turn = (board.turn == chess.WHITE) == policy_is_white

            if it_is_policy_turn:
                board_np = board_to_tensor(board)
                board_t  = torch.from_numpy(board_np).float()
                mask_t   = legal_mask_for_board(board)

                with torch.no_grad():
                    x      = board_t.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,17,8,8)
                    latent = encoder(x)
                    logits, val = decoder(latent)

                masked = logits[0].masked_fill(~mask_t.to(device), float('-inf'))
                if temperature != 1.0:
                    masked = masked / temperature
                probs  = F.softmax(masked, dim=-1)
                dist   = torch.distributions.Categorical(probs)
                action = dist.sample()
                lp     = dist.log_prob(action).item()
                value  = val.item()
                action = action.item()

                move = idx_to_move(board, action)
                if move is None or move not in board.legal_moves:
                    move   = random.choice(list(board.legal_moves))
                    action = move_to_idx(move)

                board.push(move)
                done = board.is_game_over()
                reward = 0.0
                if done:
                    result = board.result()
                    if result == "1-0":
                        reward = 1.0 if policy_is_white else -1.0
                    elif result == "0-1":
                        reward = -1.0 if policy_is_white else 1.0

                episode.add(TransitionWithMask(
                    board_tensor=board_t,
                    action=action,
                    log_prob=lp,
                    value=value,
                    reward=reward,
                    done=done,
                    legal_mask=mask_t,
                ))

            else:
                if opponent == "self":
                    with torch.no_grad():
                        a_opp, _, _ = select_action(board, encoder, decoder, device, temperature)
                    opp_move = idx_to_move(board, a_opp) if a_opp is not None else None
                    if opp_move not in board.legal_moves:
                        opp_move = None
                else:
                    opp_move = None

                if opp_move is None:
                    opp_move = random.choice(list(board.legal_moves))

                board.push(opp_move)
                if board.is_game_over() and len(episode.transitions) > 0:
                    result = board.result()
                    if result == "1-0":
                        r = 1.0 if policy_is_white else -1.0
                    elif result == "0-1":
                        r = -1.0 if policy_is_white else 1.0
                    else:
                        r = 0.0
                    last = episode.transitions[-1]
                    episode.transitions[-1] = TransitionWithMask(
                        board_tensor=last.board_tensor,
                        action=last.action,
                        log_prob=last.log_prob,
                        value=last.value,
                        reward=r,
                        done=True,
                        legal_mask=last.legal_mask,
                    )
                    break

        episodes.append(episode)

    return episodes


def ppo_update_with_masks(
    encoder, decoder, optimizer, episodes, device,
    clip_eps, value_coef, entropy_coef, grad_clip,
    ppo_epochs, minibatch_size, gamma, lam, amp_ctx, scaler,
) -> dict:
    all_boards, all_actions, all_old_lp, all_masks = [], [], [], []
    all_advantages, all_returns = [], []

    for ep in episodes:
        if len(ep) == 0:
            continue
        rewards = [t.reward  for t in ep.transitions]
        values  = [t.value   for t in ep.transitions]
        dones   = [t.done    for t in ep.transitions]
        advs, rets = compute_gae(rewards, values, dones, gamma, lam)

        for t, adv, ret in zip(ep.transitions, advs, rets):
            all_boards.append(t.board_tensor)
            all_actions.append(t.action)
            all_old_lp.append(t.log_prob)
            all_masks.append(t.legal_mask)
            all_advantages.append(adv)
            all_returns.append(ret)

    if len(all_boards) == 0:
        return {}

    boards_t  = torch.stack(all_boards).to(device)
    actions_t = torch.tensor(all_actions, dtype=torch.long,    device=device)
    old_lp_t  = torch.tensor(all_old_lp,  dtype=torch.float32, device=device)
    masks_t   = torch.stack(all_masks).to(device)
    advs_t    = torch.tensor(all_advantages, dtype=torch.float32, device=device)
    rets_t    = torch.tensor(all_returns,    dtype=torch.float32, device=device)

    advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

    N = boards_t.shape[0]
    total_loss = total_policy = total_value = total_entropy = 0.0
    n_updates  = 0

    decoder.train()
    for _ in range(ppo_epochs):
        perm = torch.randperm(N, device=device)
        for start in range(0, N, minibatch_size):
            idx = perm[start : start + minibatch_size]

            mb_boards  = boards_t[idx]
            mb_actions = actions_t[idx]
            mb_old_lp  = old_lp_t[idx]
            mb_masks   = masks_t[idx]
            mb_advs    = advs_t[idx]
            mb_rets    = rets_t[idx]

            optimizer.zero_grad()
            with amp_ctx():
                new_lp, new_val = compute_log_prob(mb_boards, mb_actions, mb_masks, encoder, decoder)

                ratio = (new_lp - mb_old_lp).exp()
                surr1 = ratio * mb_advs
                surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(new_val, mb_rets)

                # Entropy (recompute logits for this minibatch)
                x_in    = mb_boards.unsqueeze(1)
                latent  = encoder(x_in)
                logits, _ = decoder(latent)
                masked  = logits.masked_fill(~mb_masks, float('-inf'))
                probs   = F.softmax(masked, dim=-1).clamp(min=1e-8)
                log_p   = probs.log()
                entropy = -(probs * log_p).sum(dim=-1).mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
                optimizer.step()

            total_loss    += loss.item()
            total_policy  += policy_loss.item()
            total_value   += value_loss.item()
            total_entropy += entropy.item()
            n_updates     += 1

    return {
        "loss":          total_loss    / n_updates,
        "policy":        total_policy  / n_updates,
        "value":         total_value   / n_updates,
        "entropy":       total_entropy / n_updates,
        "n_transitions": N,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO RL fine-tuning of TransformerMoveDecoder")
    parser.add_argument("--jepa_ckpt",      required=True,  help="Path to JEPA checkpoint")
    parser.add_argument("--decoder_ckpt",   default=None,   help="Supervised decoder checkpoint to warm-start from")
    parser.add_argument("--out",            default="best_move/decoder_ppo.pt")
    parser.add_argument("--n_iterations",   type=int,   default=500)
    parser.add_argument("--games_per_iter", type=int,   default=32,  help="Games per iteration")
    parser.add_argument("--max_moves",      type=int,   default=200, help="Max half-moves per game")
    parser.add_argument("--opponent",       default="random", choices=["random", "self"])
    parser.add_argument("--temperature",    type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--gamma",          type=float, default=0.99)
    parser.add_argument("--lam",            type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--ppo_epochs",     type=int,   default=4)
    parser.add_argument("--minibatch",      type=int,   default=256)
    parser.add_argument("--clip_eps",       type=float, default=0.2)
    parser.add_argument("--value_coef",     type=float, default=0.5)
    parser.add_argument("--entropy_coef",   type=float, default=0.01)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--grad_clip",      type=float, default=0.5)
    parser.add_argument("--log_every",      type=int,   default=10)
    args = parser.parse_args()

    train_ppo(
        jepa_ckpt=args.jepa_ckpt,
        decoder_ckpt=args.decoder_ckpt,
        n_iterations=args.n_iterations,
        games_per_iter=args.games_per_iter,
        max_moves=args.max_moves,
        opponent=args.opponent,
        temperature=args.temperature,
        gamma=args.gamma,
        lam=args.lam,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        grad_clip=args.grad_clip,
        output_path=args.out,
        log_every=args.log_every,
    )
