# chessjepa

This repository implements a JEPA-based chess engine. The move decoder now uses a dual‑head
architecture: a policy head producing move logits and a value head predicting a scalar
evaluation (win probability / centipawn score). Training scripts support an additional mean
squared error loss for the value prediction, and the dataset generators can include Stockfish
or game-result evaluations under the `evals` key. The web GUI shows the predicted evaluation
alongside the best move.

