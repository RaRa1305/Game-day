# 2048 Strategy with Evaluation

## Overview
This program implements the 2048 game with a heuristic-based AI player that evaluates and selects the best moves. The program includes a manual mode, AI-assisted play, and a fully automated AI gameplay mode with performance evaluation.

## Features
- **Manual Gameplay:** Users can play 2048 by entering movement commands.
- **AI Assistance:** The AI can suggest the best move based on a heuristic evaluation.
- **Fully Automated AI Play:** The AI plays the game autonomously and evaluates its performance.
- **Performance Evaluation:** The AI tracks and evaluates its performance based on metrics such as max tile, score per move, and board organization quality.

## Installation
### Requirements:
- Python 3.x
- NumPy

### Steps:
1. Clone or download the repository.
2. Install NumPy if not already installed:
   ```bash
   pip install numpy
   ```
3. Run the script:
   ```bash
   python 2048_Game.py
   ```

## How to Play
Upon running the script, the user will be prompted with three options:
1. **Manual Mode:** Play the game manually by entering movement commands.
   - Use `R` for right, `L` for left, `U` for up, and `D` for down.
   - Type `AI` at any move to let the AI suggest a move.
2. **View Controls:** Displays available commands.
3. **AI Auto-Play Mode:** The AI plays the game and evaluates its performance.

## AI Strategy
The AI evaluates the board based on the following heuristics:
- **Empty Cells:** More empty cells provide better opportunities for merging.
- **Monotonicity:** Encourages a decreasing or increasing order of tile values.
- **Smoothness:** Prefers similar adjacent tiles to facilitate merging.
- **Corner Maximum:** Prefers placing the highest value tile in a corner.

The AI selects the move that maximizes the overall board evaluation score.

## Evaluation Metrics
At the end of an AI-play session, the program provides a performance report including:
- Maximum tile achieved
- Total score
- Number of moves played
- Score per move efficiency
- Board organization quality
- Overall rating

## Author
This program was made by Rakshit Ranka to test and implement strategies for 2048.

## License
This project is open-source and available for modification and redistribution.

