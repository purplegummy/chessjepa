document.addEventListener('DOMContentLoaded', () => {
    
    // UI Elements
    const jepaMeterFill = document.getElementById('meterFill');
    const sfMeterFill = document.getElementById('sfMeterFill');
    const sfEvalBadge = document.getElementById('sfEvalBadge');
    const whiteProbVal = document.getElementById('whiteProbVal');
    const blackProbVal = document.getElementById('blackProbVal');
    const evalStatus = document.getElementById('evalStatus');
    const fenInput = document.getElementById('fenInput');
    // Chess logic
    const game = new Chess();
    let evaluationTimeout = null;

    // Evaluate current FEN via backend API
    async function evaluatePosition(fen) {
        // Debounce requests slightly to avoid spamming the backend during dragging
        if (evaluationTimeout) clearTimeout(evaluationTimeout);
        
        evaluationTimeout = setTimeout(async () => {
            evalStatus.textContent = 'Evaluating...';
            evalStatus.classList.add('loading');
            
            try {
                const response = await fetch('/api/evaluate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ fen: fen })
                });
                
                if (!response.ok) throw new Error("Server error");
                
                const data = await response.json();
                updateUI(data, fen);
                
            } catch (error) {
                console.error("Evaluation error:", error);
                evalStatus.textContent = 'Error';
                evalStatus.classList.remove('loading');
                evalStatus.style.color = 'red';
            }
        }, 100);
    }
    
    // Update the visual meters
    function updateUI(data, fen) {
        if (!data) return;
        
        // --- JEPA Updates ---
        const jepaWhiteProb = data.jepa_win_probability || 0.5;
        const jepaWhitePct = (jepaWhiteProb * 100).toFixed(1);
        const jepaBlackPct = ((1.0 - jepaWhiteProb) * 100).toFixed(1);
        
        whiteProbVal.textContent = `${jepaWhitePct}%`;
        blackProbVal.textContent = `${jepaBlackPct}%`;
        jepaMeterFill.style.width = `${jepaWhitePct}%`;
        
        if (jepaWhiteProb > 0.6) {
            jepaMeterFill.style.boxShadow = "0 0 15px rgba(230, 237, 243, 0.8)";
        } else if (jepaWhiteProb < 0.4) {
            jepaMeterFill.style.boxShadow = "none";
        } else {
            jepaMeterFill.style.boxShadow = "0 0 5px rgba(255, 255, 255, 0.3)";
        }
        
        // --- Stockfish Updates ---
        const sfWhiteProb = data.stockfish_win_probability;
        if (sfWhiteProb !== null && sfWhiteProb !== undefined) {
            const sfWhitePct = (sfWhiteProb * 100).toFixed(1);
            sfMeterFill.style.width = `${sfWhitePct}%`;
            
            if (sfWhiteProb > 0.6) {
                sfMeterFill.style.boxShadow = "0 0 15px rgba(230, 237, 243, 0.8)";
            } else if (sfWhiteProb < 0.4) {
                sfMeterFill.style.boxShadow = "none";
            } else {
                sfMeterFill.style.boxShadow = "0 0 5px rgba(255, 255, 255, 0.3)";
            }
        } else {
            sfMeterFill.style.width = "50%";
            sfMeterFill.style.boxShadow = "0 0 5px rgba(255, 255, 255, 0.3)";
        }
        
        // Update SF Badge
        if (data.stockfish_mate !== null) {
            sfEvalBadge.textContent = `M${data.stockfish_mate > 0 ? '+' : ''}${data.stockfish_mate}`;
            sfEvalBadge.style.color = data.stockfish_mate > 0 ? "var(--white-side)" : "#a3b1c6";
        } else if (data.stockfish_cp !== null) {
            const cp = data.stockfish_cp;
            sfEvalBadge.textContent = `${cp > 0 ? '+' : ''}${cp.toFixed(2)}`;
            sfEvalBadge.style.color = cp > 0 ? "var(--white-side)" : "#a3b1c6";
        } else {
            sfEvalBadge.textContent = "-";
            sfEvalBadge.style.color = "var(--text-muted)";
        }

        evalStatus.textContent = 'Current';
        evalStatus.classList.remove('loading');
        evalStatus.style.color = 'var(--accent)';
        
        fenInput.value = fen;
    }

    // Chessboard.js Configuration
    const boardConfig = {
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
        position: 'start',
        draggable: true,
        dropOffBoard: 'trash', // we allow removing pieces to test distinct positions!
        sparePieces: true,     // we allow dropping spare pieces!
        
        onDrop: function(source, target, piece, newPos, oldPos, orientation) {
            // Need to wait 1 tick for the DOM board state to settle
            setTimeout(() => {
                const currentFen = board.fen() + ' w - - 0 1'; // append dummy state for valid fen
                evaluatePosition(currentFen);
            }, 50);
        },
        onSnapEnd: function() {
            // Fired after piece snap animation
        }
    };

    const board = Chessboard('board', boardConfig);
    
    // Initial evaluation
    const initialFen = board.fen() + ' w KQkq - 0 1';
    fenInput.value = initialFen;
    evaluatePosition(initialFen);

    // Button Listeners
    document.getElementById('startBtn').addEventListener('click', () => {
        board.start();
        const fen = board.fen() + ' w KQkq - 0 1';
        evaluatePosition(fen);
    });

    document.getElementById('clearBtn').addEventListener('click', () => {
        board.clear();
        jepaMeterFill.style.width = `50%`;
        sfMeterFill.style.width = `50%`;
        whiteProbVal.textContent = `-`;
        blackProbVal.textContent = `-`;
        fenInput.value = '8/8/8/8/8/8/8/8 w - - 0 1';
        evalStatus.textContent = 'Empty';
    });
    
    // Generate Random FEN helper
    function generateRandomFEN() {
        // Clear board
        game.clear();
        
        const pieces = ['p', 'n', 'b', 'r', 'q'];
        const rows = ['1', '2', '3', '4', '5', '6', '7', '8'];
        const cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
        
        let allSquares = [];
        for (let r of rows) {
            for (let c of cols) {
                allSquares.push(c + r);
            }
        }
        
        // Shuffle squares
        for (let i = allSquares.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [allSquares[i], allSquares[j]] = [allSquares[j], allSquares[i]];
        }
        
        // Always place kings
        const wkSquare = allSquares.pop();
        const bkSquare = allSquares.pop();
        game.put({ type: 'k', color: 'w' }, wkSquare);
        game.put({ type: 'k', color: 'b' }, bkSquare);
        
        // Generate a random number of pieces (between 5 and 20)
        const numPieces = 5 + Math.floor(Math.random() * 16);
        for (let i = 0; i < numPieces && allSquares.length > 0; i++) {
            const sq = allSquares.pop();
            const type = pieces[Math.floor(Math.random() * pieces.length)];
            const color = Math.random() < 0.5 ? 'w' : 'b';
            game.put({ type: type, color: color }, sq);
        }
        
        return game.fen();
    }
    
    document.getElementById('randomBtn').addEventListener('click', () => {
        const fen = generateRandomFEN();
        board.position(fen);
        evaluatePosition(fen);
    });
});
