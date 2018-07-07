import numpy as np


def valid_actions(game_board):
    actions = []
    count = 0
    state_size = len(game_board)

    for i in range(state_size):
        for j in range(state_size):
            if game_board[i][j] == 0:
                actions.append([(i, j), count])
            count += 1

    return actions


# Check win
def check_win(game_board, win_mark):
    num_mark = np.count_nonzero(game_board)
    state_size = len(game_board)
    # Check four stones in a row (Horizontal)
    for row in range(state_size):
        for col in range(state_size - win_mark + 1):
            # Black win!
            if np.sum(game_board[row, col:col + win_mark]) == win_mark:
                return 1
            # White win!
            if np.sum(game_board[row, col:col + win_mark]) == -win_mark:
                return 2

    # Check four stones in a colum (Vertical)
    for row in range(state_size - win_mark + 1):
        for col in range(state_size):
            # Black win!
            if np.sum(game_board[row: row + win_mark, col]) == win_mark:
                return 1
            # White win!
            if np.sum(game_board[row: row + win_mark, col]) == -win_mark:
                return 2

    # Check four stones in diagonal (Diagonal)
    for row in range(state_size - win_mark + 1):
        for col in range(state_size - win_mark + 1):
            count_sum = 0
            for i in range(win_mark):
                if game_board[row + i, col + i] == 1:
                    count_sum += 1
                if game_board[row + i, col + i] == -1:
                    count_sum -= 1

            # Black Win!
            if count_sum == win_mark:
                return 1

            # White WIN!
            if count_sum == -win_mark:
                return 2

    for row in range(win_mark - 1, state_size):
        for col in range(state_size - win_mark + 1):
            count_sum = 0
            for i in range(win_mark):
                if game_board[row - i, col + i] == 1:
                    count_sum += 1
                if game_board[row - i, col + i] == -1:
                    count_sum -= 1

            # Black Win!
            if count_sum == win_mark:
                return 1

            # White WIN!
            if count_sum == -win_mark:
                return 2

    # Draw (board is full)
    if num_mark == state_size * state_size:
        return 3

    # If No winner or no draw
    return 0