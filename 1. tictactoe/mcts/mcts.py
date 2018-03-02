from utils import valid_actions, check_win
from copy import deepcopy
import numpy as np
import random
import time

import tictactoe as game


class MCTS:
    def __init__(self, state_size, action_size, win_mark):
        # Get parameters
        self.state_size = state_size
        self.action_size = action_size
        self.win_mark = win_mark

    def selection(self, tree):
        node_id = (0,)

        while True:
            # Check if current node is leaf node
            if len(tree[node_id]['child']) == 0:
                # print(len(tree[node_id]['child']))
                return node_id
            else:
                max_value = -100
                parent_id = node_id
                for i in range(len(tree[node_id]['child'])):
                    child_id = parent_id + (tree[parent_id]['child'][i],)
                    current_w = tree[child_id]['w']
                    current_n = deepcopy(tree[child_id]['n'])
                    total_n = tree[tree[child_id]['parent']]['n']

                    if current_n == 0:
                        current_n = 0.000001

                    q = current_w / current_n
                    u = 10 * np.sqrt(2 * np.log(total_n) / current_n)

                    if q + u > max_value:
                        max_value = q + u
                        node_id = child_id

    def expansion(self, tree, leaf_id):
        leaf_state = deepcopy(tree[leaf_id]['state'])
        num_mark = np.count_nonzero(leaf_state)
        is_terminal = check_win(leaf_state,
                                num_mark,
                                self.win_mark)
        actions = valid_actions(leaf_state)
        expand_thres = 20

        if leaf_id == (0,) or tree[leaf_id]['n'] > expand_thres:
            is_expand = True
        else:
            is_expand = False

        if is_terminal == 0 and is_expand:
            # expansion for every possible actions
            childs = []
            for action in actions:
                state = deepcopy(tree[leaf_id]['state'])
                chosen_index = action[1]
                current_player = tree[leaf_id]['player']

                if current_player == 0:
                    next_turn = 1
                    state[action[0]] = 1
                else:
                    next_turn = 0
                    state[action[0]] = -1

                child_id = leaf_id + (chosen_index, )
                childs.append(child_id)
                tree[child_id] = {'state': state,
                                  'player': next_turn,
                                  'child': [],
                                  'parent': leaf_id,
                                  'n': 0,
                                  'w': 0,
                                  'q': 0}

                tree[leaf_id]['child'].append(chosen_index)

            child_id = random.sample(childs, 1)
            return tree, child_id[0]
        else:
            # If leaf node is terminal state, just return MCTS tree and True
            return tree, leaf_id

    def simulation(self, tree, child_id):
        state = deepcopy(tree[child_id]['state'])
        player = deepcopy(tree[child_id]['player'])

        while True:
            win = check_win(state,
                            np.count_nonzero(state),
                            self.win_mark)
            if win != 0:
                return win
            else:
                actions = valid_actions(state)
                action = random.choice(actions)
                if player == 0:
                    player = 1
                    state[action[0]] = 1
                else:
                    player = 0
                    state[action[0]] = -1

    def backup(self, tree, child_id, sim_result):
        current_player = deepcopy(tree[(0,)]['player'])
        current_id = child_id

        if sim_result == 3:
            value = 1
        elif sim_result - 1 == current_player:
            value = 1
        else:
            value = 0

        while True:
            tree[current_id]['n'] += 1
            tree[current_id]['w'] += value
            tree[current_id]['q'] = tree[current_id]['w'] / \
                                    tree[current_id]['n']
            tree[tree[current_id]['parent']]['n'] += 1

            if tree[current_id]['parent'] == (0,):
                return tree
            else:
                current_id = tree[current_id]['parent']


if __name__ == '__main__':
    # tic-tac-toe game environment
    env = game.GameState()
    state_size, win_mark = game.Return_BoardParams()
    action_size = game.Return_Num_Action()

    agent = MCTS(state_size, action_size, win_mark)

    board_shape = [state_size, state_size]
    game_board = np.zeros(board_shape)

    do_mcts = True
    num_mcts = 2000
    # 0: Black, 1: White
    turn = 0

    while True:
        # Select action
        action = 0

        # MCTS
        if do_mcts:
            # Initialize Tree
            root_id = (0,)
            tree = {root_id: {'state': game_board,
                              'player': turn,
                              'child': [],
                              'parent': None,
                              'n': 0,
                              'w': None,
                              'q': None}}

            count = 0
            for i in range(num_mcts):
                # step 1: selection
                leaf_id = agent.selection(tree)
                # step 2: expansion
                tree, child_id = agent.expansion(tree, leaf_id)
                # step 3: simulation
                # sim_result: 1 = O win, 2 = X win, 3 = Draw
                sim_result = agent.simulation(tree, child_id)
                # step 4: backup
                tree = agent.backup(tree, child_id, sim_result)
                count += 1

            print('-------- current state --------')
            print(tree[(0,)]['state'])
            Q_list = {}
            for i in tree[(0,)]['child']:
                Q_list[(0, i)] = tree[(0, i)]['q']

            # Find Max Action
            max_action = max(Q_list, key=Q_list.get)[1]
            print('Max Action: ' + str(max_action + 1))
            do_mcts = False

        # Take action and get info. for update
        game_board, check_valid_pos, win_index, turn = env.step(action)

        # If one move is done
        if check_valid_pos:
            do_mcts = True

        # If game is finished
        if win_index != 0:
            do_mcts = True
            game_board = np.zeros(board_shape)

        # Delay for visualization
        time.sleep(0.01)
