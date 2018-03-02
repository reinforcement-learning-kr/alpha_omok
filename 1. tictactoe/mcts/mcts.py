import tictactoe as game
import numpy as np
import random
import time
import copy


class MCTS:
    def __init__(self, env, state_size, action_size, win_mark):
        # Get parameters
        self.state_size = state_size
        self.action_size = action_size

        # Initialize Parameters
        self.step = 1
        self.score = 0
        self.episode = 0

        # Turn = 0: Black, 1: White
        self.turn = 0
        self.check_valid_pos = 0
        self.win_index = 0

        self.win_mark = win_mark

    def selection(self, tree, edges):
        node_id = (0,)
        # Loop until finding leaf node
        while True:
            # Check if current node is leaf node
            valid_actions = self.find_legal_moves(tree[node_id][''])
            if len(tree[node_id]['child']) == 0:
                # Leaf node!
                return node_id
            else:
                Max_QU = -100
                parent_id = node_id
                for i in range(len(tree[node_id]['child'])):
                    id_temp = parent_id + (tree[parent_id]['child'][i],)
                    current_w = edges[id_temp]['W']
                    current_n = copy.deepcopy(edges[id_temp]['N'])
                    parent_n = tree[edges[id_temp]['parent_node']][
                        'total_n']

                    if current_n == 0:
                        current_n = 0.000001

                    Q = current_w / current_n
                    U = 10 * np.sqrt(2 * np.log(parent_n) / current_n)

                    if Q + U > Max_QU:
                        Max_QU = Q + U
                        node_id = id_temp

    def expansion(self, tree, edges, leaf_id):
        # Find legal move
        current_board = copy.deepcopy(tree[leaf_id]['state'])
        is_terminal = self.check_win(current_board,
                                     np.count_nonzero(current_board))
        legal_moves = self.find_legal_moves(current_board)
        expand_thres = 5

        if leaf_id == (0,) or tree[leaf_id]['total_n'] > expand_thres:
            is_expand = True
        else:
            is_expand = False

        if len(legal_moves) > 0 and is_terminal == 0 and is_expand:
            for legal_move in legal_moves:
                # Initialize current board at every legal move
                current_board = copy.deepcopy(tree[leaf_id]['state'])

                chosen_coord = legal_move[0]
                chosen_index = legal_move[1]

                current_player = tree[leaf_id]['player']

                if current_player == 0:
                    next_turn = 1
                    current_board[chosen_coord[0]][chosen_coord[1]] = 1
                else:
                    next_turn = 0
                    current_board[chosen_coord[0]][chosen_coord[1]] = -1

                child_id = leaf_id + (chosen_index,)
                tree[child_id] = {'state': current_board,
                                  'player': next_turn,
                                  'child': [],
                                  'parent': leaf_id,
                                  'total_n': 0}

                edges[child_id] = {'N': 0, 'W': 0, 'Q': 0,
                                   'parent_node': leaf_id}

                tree[leaf_id]['child'].append(chosen_index)

            return tree, edges, child_id
        else:
            # If leaf node is terminal state, just return MCTS tree and True
            return tree, edges, leaf_id

    def simulation(self, tree, edges, child_id):
        current_board = copy.deepcopy(tree[child_id]['state'])
        current_player = copy.deepcopy(tree[child_id]['player'])
        while True:
            if self.check_win(current_board,
                              np.count_nonzero(current_board)) != 0:
                return self.check_win(current_board,
                                      np.count_nonzero(current_board))
            else:
                legal_moves = self.find_legal_moves(current_board)

                chosen_move = random.choice(legal_moves)
                chosen_coord = chosen_move[0]
                # chosen_index = chosen_move[1]

                if current_player == 0:
                    current_player = 1
                    current_board[chosen_coord[0]][chosen_coord[1]] = 1
                else:
                    current_player = 0
                    current_board[chosen_coord[0]][chosen_coord[1]] = -1

    def backup(self, tree, edges, child_id, sim_result):
        current_player = copy.deepcopy(tree[(0,)]['player'])
        current_id = child_id

        if sim_result == 3:
            value = 0.7
        elif sim_result - 1 == current_player:
            value = 1
        else:
            value = -1

        while True:
            edges[current_id]['N'] += 1
            edges[current_id]['W'] += value
            edges[current_id]['Q'] = edges[current_id]['W'] / \
                                        edges[current_id]['N']
            tree[edges[current_id]['parent_node']]['total_n'] += 1

            if tree[current_id]['parent'] == (0,):
                return tree, edges
            else:
                current_id = tree[current_id]['parent']

    def find_legal_moves(self, game_board):
        legal_moves = []
        count_moves = 0
        for i in range(self.state_size):
            for j in range(self.state_size):
                if game_board[i][j] == 0:
                    legal_moves.append([(i, j), count_moves])
                count_moves += 1
        return legal_moves

    # Check win
    def check_win(self, game_board, num_mark):
        # Check four stones in a row (Horizontal)
        for row in range(self.state_size):
            for col in range(self.state_size - self.win_mark + 1):
                # Black win!
                if np.sum(game_board[row,
                          col:col + self.win_mark]) == self.win_mark:
                    return 1
                # White win!
                if np.sum(game_board[row,
                          col:col + self.win_mark]) == -self.win_mark:
                    return 2

        # Check four stones in a colum (Vertical)
        for row in range(self.state_size - self.win_mark + 1):
            for col in range(self.state_size):
                # Black win!
                if np.sum(game_board[row: row + self.win_mark,
                          col]) == self.win_mark:
                    return 1
                # White win!
                if np.sum(game_board[row: row + self.win_mark,
                          col]) == -self.win_mark:
                    return 2

        # Check four stones in diagonal (Diagonal)
        for row in range(self.state_size - self.win_mark + 1):
            for col in range(self.state_size - self.win_mark + 1):
                count_sum = 0
                for i in range(self.win_mark):
                    if game_board[row + i, col + i] == 1:
                        count_sum += 1
                    if game_board[row + i, col + i] == -1:
                        count_sum -= 1

                # Black Win!
                if count_sum == self.win_mark:
                    return 1

                # White WIN!
                if count_sum == -self.win_mark:
                    return 2

        for row in range(self.win_mark - 1, self.state_size):
            for col in range(self.state_size - self.win_mark + 1):
                count_sum = 0
                for i in range(self.win_mark):
                    if game_board[row - i, col + i] == 1:
                        count_sum += 1
                    if game_board[row - i, col + i] == -1:
                        count_sum -= 1

                # Black Win!
                if count_sum == self.win_mark:
                    return 1

                # White WIN!
                if count_sum == -self.win_mark:
                    return 2

        # Draw (board is full)
        if num_mark == self.state_size * self.state_size:
            return 3

        # If No winner or no draw
        return 0


if __name__ == '__main__':
    # tic-tac-toe game environment
    env = game
    state_size, win_mark = game.Return_BoardParams()
    action_size = env.Return_Num_Action()

    agent = MCTS(env, state_size, action_size, win_mark)
    env_step = env.GameState()
    board_shape = [state_size, state_size]
    game_board = np.zeros(board_shape)

    do_mcts = True
    num_mcts = 1000

    while True:
        # Select action
        action = 0

        # MCTS
        if do_mcts:
            start_time = time.time()
            # Initialize Tree
            tree = {(0,): {'state': game_board, 'player': agent.turn,
                           'child': [], 'parent': None, 'total_n': 0}}
            edges = {}

            count = 0
            for i in range(num_mcts):
                leaf_id = agent.selection(tree, edges)
                tree, edges, child_id = agent.expansion(tree, edges, leaf_id)
                # sim_result: 1 = O win, 2 = X win, 3 = Draw
                sim_result = agent.simulation(tree, edges, child_id)
                tree, edges = agent.backup(tree, edges, child_id, sim_result)
                count += 1

            print('=================================')
            print(tree[(0,)]['state'])

            print(' Root Node ========================')
            print(tree[(0,)])

            print(' Edge ')
            Q_list = {}
            for i in tree[(0,)]['child']:
                print('Edge_id: ' + str([0, i]))
                print('Edge Value: ' + str(edges[(0, i)]))
                Q_list[(0, i)] = edges[(0, i)]['Q']

            # Find Max Action
            max_action = max(Q_list, key=Q_list.get)[1]
            print('Max Action: ' + str(max_action + 1))
            do_mcts = False
            print('MCTS Calculation time: ' + str(time.time() - start_time))

        # Take action and get info. for update
        game_board, agent.check_valid_pos, agent.win_index, agent.turn = \
            env_step.frame_step(action)

        # If one move is done
        if agent.check_valid_pos:
            do_mcts = True

        # If game is finished
        if agent.win_index != 0:
            do_mcts = True
            game_board = np.zeros(board_shape)

        # Delay for visualization
        time.sleep(0.01)
