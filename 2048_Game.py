import random
import numpy as np

# Basic game implementation

def gen():
    return random.randint(0, 3) # Random generator to generate the coordinates of the tile on which new number ( 2 or 4) will appear.

def start_game():
    K = np.zeros(shape=(4, 4), dtype=int)

    for _ in range(2):  
        while True:
            x, y = gen(), gen()
            if K[x][y] == 0: 
                K[x][y] = 2 # The game starts with 2 random blocks having the number 2 , all others are 0
                break

    print(K)
    return K

score = 0 # Score is incremented by the value of a newly formed tile made by merging.

def R(K):
    global score # score is implemented a s a global variable.
    def collect_right(K):
        for i in range(3,-1,-1):
            for j in range(3,-1,-1):
                if K[i][j]!=0:
                    continue
                else:
                    for k in range(j-1,-1,-1):
                        if K[i][k]==0:
                            continue
                        else:
                            K[i][j],K[i][k]=K[i][k],K[i][j]
                            break

    def combine_right(K):
        global score
        for i in range(3,-1,-1):
            for j in range(3,0,-1):
                if K[i][j]==0:
                    break
                elif K[i][j]==K[i][j-1]:
                    K[i][j]=2*K[i][j]
                    K[i][j-1]=0
                    score=score+K[i][j]
                else :
                    continue
                collect_right(K)

    previous_K=np.copy(K)

    collect_right(K)
    combine_right(K)

    if np.array_equal(previous_K,K): # If the move is invalid, no change in board will be seen , hence the player will be prompted to put a different input.
        return

    
    while True:
        x, y = gen(), gen() # Generation of new random tile.
        if K[x][y] == 0: 
            if random.randint(0,9)<9: # Chance of 2 is 90% and 4 is 10%.
                K[x][y] = 2 
            else:
                K[x][y] = 4
            break

    print(K)
    print(score)

def L(K):
    global score
    def collect_left(K):
        for i in range(0,4):
            for j in range(0,4):
                if K[i][j]!=0:
                    continue
                else:
                    for k in range(j+1,4):
                        if K[i][k]==0:
                            continue
                        else:
                            K[i][j],K[i][k]=K[i][k],K[i][j]
                            break

    def combine_left(K):
        global score
        for i in range(0,4):
            for j in range(0,3):
                if K[i][j]==0:
                    break
                elif K[i][j]==K[i][j+1]:
                    K[i][j]=2*K[i][j]
                    K[i][j+1]=0
                    score=score+K[i][j]
                else :
                    continue
                collect_left(K)

    previous_K=np.copy(K)

    collect_left(K)
    combine_left(K)

    if np.array_equal(previous_K,K):
        return

    
    while True:
        x, y = gen(), gen()
        if K[x][y] == 0: 
            if random.randint(0,9)<9:
                K[x][y] = 2 
            else:
                K[x][y] = 4
            break

    print(K)
    print(score)

def D(K):
    global score
    def collect_down(K):
        for j in range(3,-1,-1):
            for i in range(3,-1,-1):
                if K[i][j]!=0:
                    continue
                else:
                    for k in range(i-1,-1,-1):
                        if K[k][j]==0:
                            continue
                        else:
                            K[i][j],K[k][j]=K[k][j],K[i][j]
                            break

    def combine_down(K):
        global score
        for j in range(3,-1,-1):
            for i in range(3,0,-1):
                if K[i][j]==0:
                    break
                elif K[i][j]==K[i-1][j]:
                    K[i][j]=2*K[i][j]
                    K[i-1][j]=0
                    score=score+K[i][j]
                else :
                    continue
                collect_down(K)

    previous_K=np.copy(K)

    collect_down(K)
    combine_down(K)

    if np.array_equal(previous_K,K):
        return

    
    while True:
        x, y = gen(), gen()
        if K[x][y] == 0: 
            if random.randint(0,9)<9:
                K[x][y] = 2 
            else:
                K[x][y] = 4
            break

    print(K)
    print(score)

def U(K):
    global score
    def collect_up(K):
        for j in range(0,4):
            for i in range(0,4):
                if K[i][j]!=0:
                    continue
                else:
                    for k in range(i+1,4):
                        if K[k][j]==0:
                            continue
                        else:
                            K[i][j],K[k][j]=K[k][j],K[i][j]
                            break

    def combine_up(K):
        global score
        for j in range(0,4):
            for i in range(0,3):
                if K[i][j]==0:
                    break
                elif K[i][j]==K[i+1][j]:
                    K[i][j]=2*K[i][j]
                    K[i+1][j]=0
                    score=score+K[i][j]
                else :
                    continue
                collect_up(K)
    

    previous_K=np.copy(K)

    collect_up(K)
    combine_up(K)

    if np.array_equal(previous_K,K):
        return

    
    while True:
        x, y = gen(), gen()
        if K[x][y] == 0: 
            if random.randint(0,9)<9:
                K[x][y] = 2 
            else:
                K[x][y] = 4
            break

    print(K)
    print(score)

# Heuristic based strategy for solving the game.

def get_empty_cells(grid):
    # Count the number of empty cells , more the better.
    return np.count_nonzero(grid==0)

def monotonic(grid):
    # A monotonic sequence is desirable as getting a smaller number surrounded by larger numbers or vice-versa is not preferred.
    def monotonic_score(arr):
        score = 0
        for i in range(len(arr) - 1):
            if arr[i] >= arr[i+1]:
                score += 1
        return score
    
    score = 0
    # Check rows
    for i in range(4):
        score += max(monotonic_score(grid[i]), monotonic_score(grid[i][::-1]))
    
    # Check columns
    for j in range(4):
        col = grid[:, j] # Selects the jth column as a 2D array is an array of horizontal arrays(rows) , so the code is slightly different than choosing rows
        score += max(monotonic_score(col), monotonic_score(col[::-1]))
    
    return score

def smoothness(grid):
    # Smoothnes refers to the absence of sudden change in values of consecutive tiles as larger the difference, more the moves required to combine them in the future.
    smoothness_score = 0
    for i in range(4):
        for j in range(4):
            if grid[i][j] != 0:
                if j < 3 and grid[i][j+1] != 0:
                    smoothness_score -= abs(grid[i][j] - grid[i][j+1]) # Negative sign because a larger value of absolute difference is more problematic.
                if i < 3 and grid[i+1][j] != 0:
                    smoothness_score -= abs(grid[i][j] - grid[i+1][j])
    return smoothness_score

def corner_max(grid):
    # Maximum tile should be in the corner, this conclusion follows from our heuristic of smoothness and monotonicity.
    corner_score = 0
    corners = [(0,0), (0,3), (3,0), (3,3)]
    for i, j in corners:
        if grid[i][j] == np.max(grid):
            corner_score += grid[i][j]*2 
    return corner_score

def evaluate_board(grid):
    # Return a combined score by appropriately assigning weights to each metric.
    empty_weight = 10.0
    monotonic_weight = 2.0
    smoothness_weight = 1.0
    corner_weight = 5.0
    
    empty_score = get_empty_cells(grid) * empty_weight
    monotonic_score = monotonic(grid) * monotonic_weight
    smoothness_score = smoothness(grid) * smoothness_weight
    corner_score = corner_max(grid) * corner_weight
    
    return empty_score + monotonic_score + smoothness_score + corner_score

def simulate_move(grid, move_func):
    # At every point, the autoplayer will try all four moves and decide the best based on evaluate_board function.
    global score
    # Save original board for comparing to find valid move.
    original_grid = np.copy(grid)
    original_score = score
    
    # Created a copy to simulate the move.
    test_grid = np.copy(grid)
    
    # Apply move
    move_func(test_grid)
    
    # Check if move changed the board
    if np.array_equal(original_grid, test_grid):
        # Move was invalid
        score = original_score
        return test_grid, False
    else:
        # Move was valid
        score = original_score  # Reset score after simulation as the best move may have a different score.
        return test_grid, True

def ai_move(grid):
    # The autoplayer AI simulates all four moves and chooses the ebst.
    move_functions = [R, L, U, D]
    move_names = ["R", "L", "U", "D"]
    
    best_score = float('-inf')
    best_move = None
    
    for i, move_func in enumerate(move_functions):
        # Create a test grid for simulation
        test_grid = np.copy(grid)
        
        # Simulate move
        new_grid, valid_move = simulate_move(test_grid, move_func)
        
        if valid_move:
            # Evaluate the board after move
            move_score = evaluate_board(new_grid)
            
            if move_score > best_score:
                best_score = move_score
                best_move = move_names[i]
    
    return best_move

# Function to evaluate the performance of the strategy

def evaluate_strategy(grid, current_score, moves):
    max_tile = np.max(grid)
    
    # Calculate score per move as a measure of merging quality as score is from merging tiles.
    score_per_move = current_score / moves if moves > 0 else 0
    
    # Calculate board organization quality based on the metrics I have defined above.
    organization = (corner_max(grid) / max_tile if max_tile > 0 else 0) + (smoothness(grid) / -1000)
    
    performance = {
        "max_tile": max_tile,
        "score": current_score,
        "moves": moves,
        "score_per_move": score_per_move,
        "organization_quality": organization,
        "overall_rating": score_per_move / 10 + (np.log2(max_tile) if max_tile > 0 else 0) + organization # For getting one 2048 tile 16-19 is an optimal rating (from my calculations)
    }
    
    return performance

def auto_play(K):
    
    moves_played = 0
    max_tile = 0
    
    print("Starting auto play...")
    
    while True:
        # Get the current max tile
        current_max = np.max(K)
        if current_max > max_tile:
            max_tile = current_max
            print(f"New max tile: {max_tile}")
        
        # Check if game is over (no valid moves)
        valid_moves = False
        for move_func in [R, L, U, D]:
            test_grid = np.copy(K)
            _,is_valid = simulate_move(test_grid, move_func)
            if is_valid:
                valid_moves = True
                break
        
        if not valid_moves:
            print("Game over! No valid moves left.")
            print(f"Final score: {score}, Max tile: {max_tile}")
            
            # Final performance evaluation
            performance = evaluate_strategy(K, score, moves_played)

            print("\nSTRATEGY PERFORMANCE REPORT :-")
            print(f"Moves played: {moves_played}")
            print(f"Score per move: {performance['score_per_move']:.2f}")
            print(f"Highest tile: {performance['max_tile']}")
            print(f"Overall rating: {performance['overall_rating']:.1f}")
            
            break
        
        # Autoplayer provides the best move based on the score calculation.
        best_move = ai_move(K)
        print(f"AI chooses: {best_move}")
        
        # Execute the move
        if best_move == "R":
            R(K)
        elif best_move == "L":
            L(K)
        elif best_move == "U":
            U(K)
        elif best_move == "D":
            D(K)
        
        moves_played += 1
        
# Basic user interface and command panel display.

print("Welcome to 2048!\nPress 1 to start manually\nPress 2 for controls\nPress 3 for AI autoplay")
choice = int(input())

if choice == 1:
    K = start_game()
    while True:
        a = str(input())
        if a == "R":
            R(K)
        if a == "L":
            L(K)
        if a == "U":
            U(K)
        if a == "D":
            D(K)
        if a == "AI": # Added option to let autoplayer AI make a single move if player gets stuck.
            move = ai_move(K)
            print(f"AI suggests: {move}")
            if move == "R":
                R(K)
            elif move == "L":
                L(K)
            elif move == "U":
                U(K)
            elif move == "D":
                D(K)
            
elif choice == 2:
    print("R for swiping right\nL for swiping left\nU for swiping up\nD for swiping down.\nAI for letting AI make one move.")
elif choice == 3:
    K = start_game()
    auto_play(K)