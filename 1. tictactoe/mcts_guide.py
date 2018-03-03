from utils import valid_actions, check_win
from copy import deepcopy
import numpy as np
import random
import time

import env as game


class MCTS:
    def __init__(self, win_mark):
        # Get parameters
        self.win_mark = win_mark

    def selection(self, tree):
        node_id = (0,)

        while True:
            num_child = len(tree[node_id]['child'])
            # Check if current node is leaf node
            if num_child == 0:
                return node_id
            else:
                max_value = -100
                leaf_id = node_id
                for i in range(num_child):
                    action = tree[leaf_id]['child'][i]
                    child_id = leaf_id + (action,)
                    w = tree[child_id]['w']
                    n = tree[child_id]['n']
                    total_n = tree[tree[child_id]['parent']]['n']

                    # for unvisited child, cannot compute u value
                    # so make n to be very small number
                    if n == 0:
                        q = w / 0.0001
                        u = 10 * np.sqrt(2 * np.log(total_n) / 0.0001)
                    else:
                        q = w / n
                        u = 10 * np.sqrt(2 * np.log(total_n) / n)

                    if q + u > max_value:
                        max_value = q + u
                        node_id = child_id

    def expansion(self, tree, leaf_id):
        leaf_state = deepcopy(tree[leaf_id]['state'])
        is_terminal = check_win(leaf_state, self.win_mark)
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
                action_index = action[1]
                current_player = tree[leaf_id]['player']

                if current_player == 0:
                    next_turn = 1
                    state[action[0]] = 1
                else:
                    next_turn = 0
                    state[action[0]] = -1

                child_id = leaf_id + (action_index, )
                childs.append(child_id)
                tree[child_id] = {'state': state,
                                  'player': next_turn,
                                  'child': [],
                                  'parent': leaf_id,
                                  'n': 0,
                                  'w': 0,
                                  'q': 0}

                tree[leaf_id]['child'].append(action_index)

            child_id = random.sample(childs, 1)
            return tree, child_id[0]
        else:
            # If leaf node is terminal state,
            # just return MCTS tree
            return tree, leaf_id

    def simulation(self, tree, child_id):
        state = deepcopy(tree[child_id]['state'])
        player = deepcopy(tree[child_id]['player'])

        while True:
            win = check_win(state, self.win_mark)
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
        player = deepcopy(tree[(0,)]['player'])
        node_id = child_id

        # sim_result: 1 = O win, 2 = X win, 3 = Draw
        if sim_result == 3:
            value = 1
        elif sim_result - 1 == player:
            value = 1
        else:
            value = 0

        while True:
            tree[node_id]['n'] += 1
            tree[node_id]['w'] += value
            tree[node_id]['q'] = tree[node_id]['w'] / tree[node_id]['n']

            parent_id = tree[node_id]['parent']
            if parent_id == (0,):
                tree[parent_id]['n'] += 1
                return tree
            else:
                node_id = parent_id


if __name__ == '__main__':
    # tic-tac-toe game environment
    env = game.GameState()
    state_size, win_mark = game.Return_BoardParams()
    agent = MCTS(win_mark)

    board_shape = [state_size, state_size]
    game_board = np.zeros(board_shape)

    do_mcts = True
    num_mcts = 2000
    # 0: O, 1: X
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

            for i in range(num_mcts):
                # step 1: selection
                leaf_id = agent.selection(tree)
                # step 2: expansion
                tree, child_id = agent.expansion(tree, leaf_id)
                # step 3: simulation
                sim_result = agent.simulation(tree, child_id)
                # step 4: backup
                tree = agent.backup(tree, child_id, sim_result)

            print('-------- current state --------')
            print(tree[(0,)]['state'])
            q_list = {}
            actions = tree[(0,)]['child']
            for i in actions:
                q_list[(0, i)] = tree[(0, i)]['q']

            # Find Max Action
            max_action = max(q_list, key=q_list.get)[1]
            print('max action: ' + str(max_action + 1))
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
