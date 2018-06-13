__all__ = ["valid_actions", "check_win", "update_state",
           "render_str", "get_state_tf", "get_state_pt", "get_action"]
import numpy as np
from collections import deque
ALPHABET = ' A B C D E F G H I J K L M N O P Q R S'


def legal_actions(game_board):
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

    current_grid = np.zeros([win_mark, win_mark])

    # check win
    for row in range(state_size - win_mark + 1):
        for col in range(state_size - win_mark + 1):
            current_grid = game_board[row: row + win_mark, col: col + win_mark]

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
    if num_mark == state_size * state_size:
        return 3

    # If No winner or no draw
    return 0


def update_state(state, turn, x_idx, y_idx):
    state[:, :, 1:16] = state[:, :, 0:15]
    state[:, :, 0] = state[:, :, 2]
    state[y_idx, x_idx, 0] = 1
    state[:, :, 16] = turn
    state = np.int8(state)

    return state


def render_str(gameboard, GAMEBOARD_SIZE, action_index):
    if action_index is not None:
        row = action_index // GAMEBOARD_SIZE
        col = action_index % GAMEBOARD_SIZE
    count = np.count_nonzero(gameboard)
    board_str = '\n  {}\n'.format(ALPHABET[:GAMEBOARD_SIZE * 2])
    for i in range(GAMEBOARD_SIZE):
        for j in range(GAMEBOARD_SIZE):
            if j == 0:
                board_str += '{:2}'.format(i + 1)
            if gameboard[i][j] == 0:
                if count > 0:
                    if col + 1 < GAMEBOARD_SIZE:
                        if (i, j) == (row, col + 1):
                            board_str += '.'
                        else:
                            board_str += ' .'
                    else:
                        board_str += ' .'
                else:
                    board_str += ' .'
            if gameboard[i][j] == 1:
                if (i, j) == (row, col):
                    board_str += '(O)'
                elif (i, j) == (row, col + 1):
                    board_str += 'O'
                else:
                    board_str += ' O'
            if gameboard[i][j] == -1:
                if (i, j) == (row, col):
                    board_str += '(X)'
                elif (i, j) == (row, col + 1):
                    board_str += 'X'
                else:
                    board_str += ' X'
            if j == GAMEBOARD_SIZE - 1:
                board_str += ' \n'
        if i == GAMEBOARD_SIZE - 1:
            board_str += '  ' + '-' * (GAMEBOARD_SIZE - 6) + \
                '  MOVE: {:2}  '.format(count) + '-' * (GAMEBOARD_SIZE - 6)
    print(board_str)


def get_state_tf(id, turn, state_size, channel_size):
    state = np.zeros([state_size, state_size, channel_size])
    length_game = len(id)

    state_1 = np.zeros([state_size, state_size])
    state_2 = np.zeros([state_size, state_size])

    channel_idx = channel_size - 1

    for i in range(length_game):
        row_idx = int(id[i] / state_size)
        col_idx = int(id[i] % state_size)

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
                state_w[:][row, col] = 1
                history.append(state_w.copy())
                color_idx = 1

    history.append(color * color_idx)
    state = np.stack(history)
    return state


def get_action_eval(pi, board):
    # need to be fixed!! apply temperature as control exploration.
    # do not select action greedily
    action_size = len(pi)
    action = np.zeros(action_size)

    valid_action = legal_actions(board)
    valid_indicator = np.zeros(action_size)
    for i in range(len(valid_action)):
        action_idx = valid_action[i][1]
        valid_indicator[action_idx] = 1

    pi = pi * valid_indicator
    # if the sum of pi equals 0, then nan eror occurs
    if np.sum(pi) > 0:
        pi /= np.sum(pi)
    else:
        pi = valid_indicator / np.sum(valid_indicator)
    action_index = np.random.choice(action_size, p=pi)
    action[action_index] = 1
    return action, action_index


def get_action(pi):
    action_size = len(pi)
    action = np.zeros(action_size)
    action_index = np.random.choice(action_size, p=pi)
    action[action_index] = 1
    return action, action_index



def get_board(node_id, board_size):
    board = np.zeros(board_size**2)

    for i, action_index in enumerate(node_id):

        if i == 0:
            if action_index == ():
                # board is none
                return None
        else:
            if i % 2 == 1:
                board[action_index] = 1
            else:
                board[action_index] = -1

    return board.reshape(board_size, board_size)


def get_turn(node_id):
    if len(node_id) % 2 == 1:
        return 0
    else:
        return 1


def get_reward(win_index, leaf_id):
    turn = get_turn(leaf_id)

    if win_index == 1:
        if turn == 1:
            # print('leaf id: {}, '
            #       'win: black, '
            #       'reward: 1'.format(leaf_id))
            reward = 1.
        else:
            # print('leaf id: {}, '
            #       'win: black, '
            #       'reward: -1'.format(leaf_id))
            reward = -1.

    elif win_index == 2:
        if turn == 1:
            # print('leaf id: {}, '
            #       'win: white, '
            #       'reward: -1'.format(leaf_id))
            reward = -1.
        else:
            # print('leaf id: {}, '
            #       'win: white, '
            #       'reward: 1'.format(leaf_id))
            reward = 1.
    else:
        # print('leaf id: {}, '
        #       'win: draw, '
        #       'reward: 0'.format(leaf_id))
        reward = 0.

    return reward


def symmetry_choice(state, pi):
    s_shape = state.shape
    history = state[:-1]
    c = state[-1]

    pi_shape = pi.shape
    p = pi.reshape(s_shape[1], s_shape[1])

    sym_state = []
    rand_i = np.random.choice(8)

    if rand_i == 0:
        return state, pi

    elif rand_i == 1:
        for s in history:
            sym_state.append(np.rot90(s, 1))

        sym_state.append(c)
        state = np.asarray(sym_state).reshape(s_shape)

        p = np.rot90(p, 1)
        pi = p.reshape(pi_shape)

    elif rand_i == 2:
        for s in history:
            sym_state.append(np.rot90(s, 2))

        sym_state.append(c)
        state = np.asarray(sym_state).reshape(s_shape)

        p = np.rot90(p, 2)
        pi = p.reshape(pi_shape)

    elif rand_i == 3:
        for s in history:
            sym_state.append(np.rot90(s, 3))

        sym_state.append(c)
        state = np.asarray(sym_state).reshape(s_shape)

        p = np.rot90(p, 3)
        pi = p.reshape(pi_shape)

    elif rand_i == 4:
        for s in history:
            sym_state.append(np.fliplr(s))

        sym_state.append(c)
        state = np.asarray(sym_state).reshape(s_shape)

        p = np.fliplr(s)
        pi = p.reshape(pi_shape)

    elif rand_i == 5:
        for s in history:
            rot_s = np.rot90(s, 1)
            sym_state.append(np.fliplr(rot_s))

        sym_state.append(c)
        state = np.asarray(sym_state).reshape(s_shape)

        rot_p = np.rot90(p, 1)
        p = np.fliplr(rot_p)
        pi = p.reshape(pi_shape)

    elif rand_i == 6:
        for s in history:
            rot_s = np.rot90(s, 2)
            sym_state.append(np.fliplr(rot_s))

        sym_state.append(c)
        state = np.asarray(sym_state).reshape(s_shape)

        rot_p = np.rot90(p, 2)
        p = np.fliplr(rot_p)
        pi = p.reshape(pi_shape)

    elif rand_i == 7:
        for s in history:
            rot_s = np.rot90(s, 3)
            sym_state.append(np.fliplr(rot_s))

        sym_state.append(c)
        state = np.asarray(sym_state).reshape(s_shape)

        rot_p = np.rot90(p, 3)
        p = np.fliplr(rot_p)
        pi = p.reshape(pi_shape)

    return state, pi
