from collections import defaultdict
from logging import getLogger
import env_small as env
logger = getLogger(__name__)

from utils import valid_actions, check_win
from copy import deepcopy
import numpy as np
import random


class Player:
    def __init__(self, win_mark, turn, board, model):
        # Get parameters
        self.win_mark = win_mark
        self.turn = turn
        self.game_board = board
        self.root_id = (0,)
        self.num_mcts = 1000
        self.model = model
        self.tree = {self.root_id: {'state': self.game_board,
                                    'player': self.turn,
                                    'child': [],
                                    'parent': None,
                                    'n': 0,
                                    'w': None,
                                    'q': None}}

    def init_mcts(self, board, turn, model):
        self.root_id = (0,)
        self.turn = turn
        self.game_board = board
        self.model = model

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
        expand_thres = 10

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
            leaf_p, leaf_v = self.model.forward(state)
            return tree, child_id[0], leaf_v
        else:
            # If leaf node is terminal state,
            # just return MCTS tree
            return tree, leaf_id

    def backup(self, tree, child_id, sim_result):
        player = deepcopy(tree[(0,)]['player'])
        node_id = child_id

        # sim_result: 1 = O win, 2 = X win, 3 = Draw
        if sim_result == 3:
            value = 0.8
        elif sim_result - 1 == player:
            value = 1
        else:
            value = -1

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

    def mcts(self):
        for i in range(self.num_mcts):
            # step 1: selection
            leaf_id = self.selection(self.tree)
            # step 2: expansion
            self.tree, child_id, value = self.expansion(self.tree, leaf_id)
            # step 3: backup
            self.tree = self.backup(self.tree, child_id, value)

    def get_policy(self, board, turn, model):
        self.init_mcts(board, turn, model)
        self.mcts()
        my_Node = self.tree[self.root_id]
        policy = np.zeros(self.action_size)
        for action, a_s in my_Node.a.items():
            policy[action] = a_s.n

        policy /= np.sum(policy)
        return policy


'''
class Node:
    def __init__(self):
        self.a = defaultdict(Edge)
        self.sum_n = 0


class Edge:
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0


class Player:
    def __init__(self, action_size=81):
        self.replay_memory = deque()
        self.action_size = action_size
        self.model = AlphaZero(action_size)
        self.tree = self.reset()

        self.change_temperature = 20
        self.epsilon = 0.25
        self.c_puct = 5
        self.dir_alpha = 0.2

    def reset(self):
        tree = defaultdict(Node)
        return tree

    def action(self, state):
        action = np.zeros(self.action_size)
        self.tree = self.reset()
        for i in range(1000):
            self.mcts(state, is_root_node=True)

        policy = self.get_policy(env)
        policy_temp = self.apply_temperature(policy, self.change_temperature)
        action_index = int(np.random.choice(action, p=policy_temp))
        action[action_index] = 1
        return policy, action

    def mcts(self, state, is_root_node=False):
        # return z
        if env.done:
            if env.winner == Winner.draw:
                return 0
            return -1

        # expansion
        if state not in self.tree:
            leaf_p, leaf_v = self.expand_and_evaluate(state)
            # make edge!! need to be fixed
            self.tree[state].p = leaf_p
            # I'm returning everything from the POV of side to move
            return leaf_v

        else:
            # selection
            action = self.select_action_q_and_u(state, is_root_node)
            my_node = self.tree[state]
            my_edge = my_node.a[action]
            my_edge.q = my_edge.w / my_edge.n

            state, _, _, _ = env.step(action)
            leaf_v = self.mcts(state) # next move from enemy POV
            leaf_v = -leaf_v

        # backup
        my_node.sum_n += 1
        my_edge.n += 1
        my_edge.w += leaf_v
        my_edge.q = my_edge.w / my_edge.n

        return leaf_v

    def expand_and_evaluate(self, state):
        leaf_p, leaf_v = self.model.forward(state)
        return leaf_p, leaf_v

    def select_action_q_and_u(self, state, is_root_node):
        my_node = self.tree[state]

        best_s = -999
        best_a = None
        i = 0

        for action, a_s in my_node.a.items():
            policy = a_s.p
            if is_root_node:
                noise = np.random.dirichlet([self.dir_alpha] * len(my_node.a))
                policy = (1 - self.epsilon) * policy + self.epsilon * noise[i]
                i += 1
            ucb = a_s.q + self.c_puct * policy * np.sqrt(
                                                my_node.sum_n + 1) / (1 + a_s.n)
            if ucb > best_s:
                best_s = ucb
                best_a = action
        return best_a

    def apply_temperature(self, policy, turn):
        tau = np.power(self.play_config.tau_decay_rate, turn + 1)
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.labels_n)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1 / tau)
            ret /= np.sum(ret)
            return ret

    def get_policy(self, env):
        state = state_key(env)
        my_Node = self.tree[state]
        policy = np.zeros(self.action_size)
        for action, a_s in my_Node.a.items():
            policy[action] = a_s.n

        policy /= np.sum(policy)
        return policy

    def finish_game(self, z):
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]

'''
