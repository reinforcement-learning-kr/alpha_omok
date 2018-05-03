import sys
import time
from utils import check_win, get_state_pt, valid_actions
from copy import deepcopy
import torch
from torch.autograd import Variable
import numpy as np

use_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
np.set_printoptions(suppress=True)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class Player:
    def __init__(self, state_size, num_mcts, inplanes):
        self.state_size = state_size
        self.num_mcts = num_mcts
        self.inplanes = inplanes
        self.win_mark = 5
        self.alpha = 0.15
        self.turn = 0
        self.board = np.zeros([self.state_size, self.state_size])
        self.root_id = (0,)
        self.model = None
        self.tree = {self.root_id: {'board': self.board,
                                    'player': self.turn,
                                    'child': [],
                                    'parent': None,
                                    'n': 0.,
                                    'w': 0.,
                                    'q': 0.,
                                    'p': None}}

    def init_mcts(self, board, turn):
        self.turn = turn
        self.board = board

    def selection(self, tree):
        node_id = self.root_id

        while True:
            num_child = len(tree[node_id]['child'])
            # check if current node is leaf node
            if num_child == 0:
                return node_id
            else:
                leaf_id = node_id
                qu = {}
                ids = []

                if leaf_id == self.root_id:
                    noise = np.random.dirichlet(
                        self.alpha * np.ones(num_child))

                for i in range(num_child):
                    action = tree[leaf_id]['child'][i]
                    child_id = leaf_id + (action,)
                    n = tree[child_id]['n']
                    q = tree[child_id]['q']

                    if leaf_id == self.root_id:
                        p = tree[child_id]['p']
                        p = 0.75 * p + 0.25 * noise[i]
                    else:
                        p = tree[child_id]['p']

                    total_n = tree[tree[child_id]['parent']]['n'] - 1

                    u = 5. * p * np.sqrt(total_n) / (n + 1)

                    if tree[leaf_id]['player'] == 0:
                        qu[child_id] = q + u

                    else:
                        qu[child_id] = q - u

                if tree[leaf_id]['player'] == 0:
                    max_value = max(qu.values())
                    ids = [key for key, value in qu.items() if value == max_value]
                    node_id = ids[np.random.choice(len(ids))]
                else:
                    min_value = min(qu.values())
                    ids = [key for key, value in qu.items() if value == min_value]
                    node_id = ids[np.random.choice(len(ids))]

    def expansion(self, tree, leaf_id):
        leaf_board = deepcopy(tree[leaf_id]['board'])
        is_terminal = check_win(leaf_board, self.win_mark)
        actions = valid_actions(leaf_board)
        turn = tree[leaf_id]['player']
        leaf_state = get_state_pt(
            leaf_id, turn, self.state_size, self.inplanes)
        # expand_thres = 10

        # if leaf_id == (0,) or tree[leaf_id]['n'] > expand_thres:
        #     is_expand = True
        # else:
        #    is_expand = False
        is_expand = True
        state_input = Variable(Tensor([leaf_state]))
        policy, value = self.model(state_input)
        policy = np.exp(policy.data.cpu().numpy()[0])
        value = value.data.cpu().numpy()[0]

        # if leaf_id == self.root_id:
        #     noise = np.random.dirichlet(
        #         self.alpha * np.ones(len(actions)))

        if is_terminal == 0 and is_expand:
            # expansion for every possible actions
            for i, action in enumerate(actions):
                board = deepcopy(tree[leaf_id]['board'])
                action_index = action[1]
                current_player = tree[leaf_id]['player']
                prior = policy[action_index]

                # if leaf_id == self.root_id:
                #     prior = 0.75 * prior + 0.25 * noise[i]

                if current_player == 0:
                    next_turn = 1
                    board[action[0]] = 1
                else:
                    next_turn = 0
                    board[action[0]] = -1

                child_id = leaf_id + (action_index,)
                tree[child_id] = {'board': board,
                                  'player': next_turn,
                                  'child': [],
                                  'parent': leaf_id,
                                  'n': 0.,
                                  'w': 0.,
                                  'q': 0.,
                                  'p': prior}

                tree[leaf_id]['child'].append(action_index)

            return tree, value

        else:
            # If leaf node is terminal
            if is_terminal == 1:
                reward = 1.
            elif is_terminal == 2:
                reward = -1.
            else:
                reward = 0.

            return tree, reward

    def backup(self, tree, leaf_id, value):
        # player = deepcopy(tree[self.root_id]['player'])
        node_id = leaf_id

        while True:
            if node_id == self.root_id:
                tree[node_id]['n'] += 1
                return tree

            """
            if tree[node_id]['player'] == player:
                tree[node_id]['w'] -= value
            else:
                tree[node_id]['w'] -= value
            """
            tree[node_id]['n'] += 1
            tree[node_id]['w'] += value
            # print('backup:', value)
            tree[node_id]['q'] = tree[node_id]['w'] / tree[node_id]['n']
            parent_id = tree[node_id]['parent']

            node_id = parent_id

    def mcts(self):
        start = time.time()
        for i in range(self.num_mcts):
            sys.stdout.write('simulation: {}\r'.format(i + 1))
            sys.stdout.flush()
            # step 1: selection
            leaf_id = self.selection(self.tree)
            # step 2: expansion
            self.tree, value = self.expansion(self.tree, leaf_id)
            # step 3: backup
            self.tree = self.backup(self.tree, leaf_id, value)
            # Delete useless tree elements
        # for key in list(self.tree.keys()):
        #     if self.root_id != key[:len(self.root_id)]:
        #         del self.tree[key]
        finish = time.time() - start
        print("{} simulations end ({:0.0f}s)".format(i + 1, finish))

    def get_pi(self, board, turn):
        self.init_mcts(board, turn)
        self.mcts()
        root_node = self.tree[self.root_id]
        pi = np.zeros(self.state_size**2, 'float')

        for action in root_node['child']:
            child_id = self.root_id + (action,)
            pi[action] = self.tree[child_id]['n']

        pi /= (self.tree[self.root_id]['n'] - 1)  # ====== why "n" is error?
        return pi

    def reset(self):
        self.turn = 0
        self.board = np.zeros([self.state_size, self.state_size])
        self.root_id = (0,)
        self.tree = {self.root_id: {'board': self.board,
                                    'player': self.turn,
                                    'child': [],
                                    'parent': None,
                                    'n': 0.,
                                    'w': 0.,
                                    'q': 0.,
                                    'p': None}}
