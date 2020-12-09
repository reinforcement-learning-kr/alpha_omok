from collections import deque

import numpy as np

ALPHABET = ' A B C D E F G H I J K L M N O P Q R S'


def valid_actions(board):
    actions = []
    count = 0
    board_size = len(board)

    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == 0:
                actions.append([(i, j), count])
            count += 1

    return actions


def legal_actions(node_id, board_size):
    all_action = {a for a in range(board_size**2)}
    action = set(node_id[1:])
    actions = all_action - action

    return list(actions)


def check_win(board, win_mark):
    board = board.copy()
    num_mark = np.count_nonzero(board)
    board_size = len(board)
    current_grid = np.zeros([win_mark, win_mark])
    for row in range(board_size - win_mark + 1):
        for col in range(board_size - win_mark + 1):
            current_grid = board[row: row + win_mark, col: col + win_mark]
            sum_horizontal = np.sum(current_grid, axis=1)
            sum_vertical = np.sum(current_grid, axis=0)
            sum_diagonal_1 = np.sum(current_grid.diagonal())
            sum_diagonal_2 = np.sum(np.flipud(current_grid).diagonal())

            # Black wins! (Horizontal and Vertical)
            if win_mark in sum_horizontal or win_mark in sum_vertical:
                return 1
            # Black wins! (Diagonal)
            if win_mark == sum_diagonal_1 or win_mark == sum_diagonal_2:
                return 1
            # White wins! (Horizontal and Vertical)
            if -win_mark in sum_horizontal or -win_mark in sum_vertical:
                return 2
            # White wins! (Diagonal)
            if -win_mark == sum_diagonal_1 or -win_mark == sum_diagonal_2:
                return 2
    # Draw (board is full)
    if num_mark == board_size * board_size:
        return 3
    # If No winner or no draw
    return 0


def render_str(board, board_size, action_index):
    if action_index is not None:
        row = action_index // board_size
        col = action_index % board_size
    count = np.count_nonzero(board)
    board_str = '\n  {}\n'.format(ALPHABET[:board_size * 2])
    for i in range(board_size):
        for j in range(board_size):
            if j == 0:
                board_str += '{:2}'.format(i + 1)
            if board[i][j] == 0:
                if count > 0:
                    if col + 1 < board_size:
                        if (i, j) == (row, col + 1):
                            board_str += '.'
                        else:
                            board_str += ' .'
                    else:
                        board_str += ' .'
                else:
                    board_str += ' .'
            if board[i][j] == 1:
                if (i, j) == (row, col):
                    board_str += '(O)'
                elif (i, j) == (row, col + 1):
                    board_str += 'O'
                else:
                    board_str += ' O'
            if board[i][j] == -1:
                if (i, j) == (row, col):
                    board_str += '(X)'
                elif (i, j) == (row, col + 1):
                    board_str += 'X'
                else:
                    board_str += ' X'
            if j == board_size - 1:
                board_str += ' \n'
        if i == board_size - 1:
            board_str += '  ' + '-' * (board_size - 6) + \
                '  MOVE: {:2}  '.format(count) + '-' * (board_size - 6)
    print(board_str)


def get_state_tf(id, turn, board_size, channel_size):
    state = np.zeros([board_size, board_size, channel_size])
    length_game = len(id)

    state_1 = np.zeros([board_size, board_size])
    state_2 = np.zeros([board_size, board_size])

    channel_idx = channel_size - 1

    for i in range(length_game):
        row_idx = int(id[i] / board_size)
        col_idx = int(id[i] % board_size)

        if i != 0:
            if i % 2 == 0:
                state_1[row_idx, col_idx] = 1
            else:
                state_2[row_idx, col_idx] = 1

        if length_game - i < channel_size:
            channel_idx = length_game - i - 1
            if i % 2 == 0:
                state[:, :, channel_idx] = state_1
            else:
                state[:, :, channel_idx] = state_2

    if turn == 0:
        state[:, :, channel_size - 1] = 1
    else:
        state[:, :, channel_size - 1] = 0

    return state


def get_state_pt(node_id, board_size, channel_size):
    state_b = np.zeros((board_size, board_size))
    state_w = np.zeros((board_size, board_size))
    color = np.ones((board_size, board_size))
    color_idx = 1
    history = deque(
        [np.zeros((board_size, board_size)) for _ in range(channel_size)],
        maxlen=channel_size)

    for i, action_idx in enumerate(node_id):
        if i == 0:
            history.append(state_b.copy())
            history.append(state_w.copy())
        else:
            row = action_idx // board_size
            col = action_idx % board_size

            if i % 2 == 1:
                state_b[row, col] = 1
                history.append(state_b.copy())
                color_idx = 0
            else:
                state_w[row, col] = 1
                history.append(state_w.copy())
                color_idx = 1

    history.append(color * color_idx)
    state = np.stack(history)

    return state


def get_board(node_id, board_size):
    board = np.zeros(board_size**2)
    for i, action_index in enumerate(node_id[1:]):
        if i % 2 == 0:
            board[action_index] = 1
        else:
            board[action_index] = -1

    return board.reshape(board_size, board_size)


def get_turn(node_id):
    if len(node_id) % 2 == 1:
        return 0
    else:
        return 1


def get_action(pi):
    action_size = len(pi)
    action = np.zeros(action_size)
    action_index = np.random.choice(action_size, p=pi)
    action[action_index] = 1

    return action, action_index


def argmax_onehot(pi):
    action_size = len(pi)
    action = np.zeros(action_size)
    max_idx = np.argwhere(pi == pi.max())
    action_index = max_idx[np.random.choice(len(max_idx))]
    action[action_index] = 1

    return action, action_index[0]


def get_reward(win_index, leaf_id):
    turn = get_turn(leaf_id)
    if win_index == 1:
        if turn == 1:
            reward = 1.
        else:
            reward = -1.
    elif win_index == 2:
        if turn == 1:
            reward = -1.
        else:
            reward = 1.
    else:
        reward = 0.

    return reward


def augment_dataset(memory, board_size):
    aug_dataset = []
    for (s, pi, z) in memory:
        for i in range(4):
            s_rot = np.rot90(s, i, axes=(1, 2)).copy()
            pi_rot = np.rot90(pi.reshape(board_size, board_size), i)
            pi_flat = pi_rot.flatten().copy()
            aug_dataset.append((s_rot, pi_flat, z))

            s_flip = np.flip(s_rot, 2).copy()
            pi_flip = np.fliplr(pi_rot).flatten().copy()
            aug_dataset.append((s_flip, pi_flip, z))

    return aug_dataset
