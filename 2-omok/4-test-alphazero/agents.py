import sys
import time

import numpy as np
import torch
from torch.autograd import Variable

import utils


use_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


class ZeroAgent(object):

    def __init__(self, board_size, num_mcts, inplanes, noise=True):
        self.board_size = board_size
        self.num_mcts = num_mcts
        self.inplanes = inplanes
        # tictactoe and omok
        self.win_mark = 3 if board_size == 3 else 5
        self.alpha = 10 / self.board_size**2
        self.c_puct = 5
        self.noise = noise
        self.root_id = None
        self.board = None
        self.turn = None
        self.model = None
        self.tree = {}

    def reset(self):
        self.root_id = None
        self.turn = None
        self.tree.clear()

    def get_pi(self, root_id, board, turn, tau):
        self._init_mcts(root_id, board, turn)
        self._mcts(self.root_id)

        visit = np.zeros(self.board_size**2, 'float')

        for action_index in self.tree[self.root_id]['child']:
            child_id = self.root_id + (action_index,)
            visit[action_index] = self.tree[child_id]['n']

        if visit.max() > 1000:
            tau = 0.1

        pi = visit**(1 / tau)
        pi /= pi.sum()
        return pi

    def _init_mcts(self, root_id, board, turn):
        self.root_id = root_id
        self.board = board
        self.turn = turn
        self.model.eval()

        if self.root_id not in self.tree:
            # print('init root node')
            self.tree[self.root_id] = {'board': self.board,
                                       'player': self.turn,
                                       'parent': None,
                                       'child': [],
                                       'n': 0.,
                                       'w': 0.,
                                       'q': 0.,
                                       'p': 0.}
        elif self.noise:
            children = self.tree[self.root_id]['child']
            noise_probs = np.random.dirichlet(
                self.alpha * np.ones(len(children)))

            for i, action_index in enumerate(children):
                child_id = self.root_id + (action_index,)
                self.tree[child_id]['p'] = 0.75 * \
                    self.tree[child_id]['p'] + 0.25 * noise_probs[i]
                # print("add noise no expansion")

    def _mcts(self, root_id):
        # start = time.time()
        for i in range(self.num_mcts):
            sys.stdout.write('simulation: {}\r'.format(i + 1))
            sys.stdout.flush()
            leaf_id, win_index = self._selection(root_id)
            value, reward = self._expansion_evaluation(leaf_id, win_index)
            self._backup(leaf_id, value, reward)

        # finish = time.time() - start
        # print("{} simulations end ({:0.0f}s)".format(i + 1, finish))

    def _selection(self, root_id):
        node_id = root_id

        while self.tree[node_id]['n'] > 0:
            win_index = utils.check_win(
                self.tree[node_id]['board'], self.win_mark)

            if win_index != 0:
                return node_id, win_index

            qu = {}
            ids = []
            total_n = 0

            for action_idx in self.tree[node_id]['child']:
                edge_id = node_id + (action_idx,)
                n = self.tree[edge_id]['n']
                total_n += n

            for i, action_index in enumerate(self.tree[node_id]['child']):
                # total_n = self.tree[node_id]['n']
                child_id = node_id + (action_index,)
                n = self.tree[child_id]['n']
                q = self.tree[child_id]['q']
                p = self.tree[child_id]['p']
                u = self.c_puct * p * np.sqrt(total_n) / (n + 1)
                qu[child_id] = q + u

            max_value = max(qu.values())
            ids = [key for key, value in qu.items()
                   if value == max_value]
            node_id = ids[np.random.choice(len(ids))]
            # print('selectionted id:', node_id)

        win_index = utils.check_win(self.tree[node_id]['board'],
                                    self.win_mark)
        return node_id, win_index

    def _expansion_evaluation(self, leaf_id, win_index):
        leaf_board = self.tree[leaf_id]['board']
        leaf_state = utils.get_state_pt(
            leaf_id, self.board_size, self.inplanes)
        state_input = Variable(Tensor([leaf_state]))
        policy, value = self.model(state_input)
        policy = policy.data.cpu().numpy()[0]
        value = value.data.cpu().numpy()[0]

        if win_index == 0:
            # print("expansion")
            actions = utils.legal_actions(leaf_board)
            prior_prob = np.zeros(self.board_size**2)

            for action in actions:
                action_index = action[1]
                prior_prob[action_index] = policy[action_index]

            prior_prob /= prior_prob.sum()

            if self.noise:
                if leaf_id == self.root_id:
                    noise_probs = np.random.dirichlet(
                        self.alpha * np.ones(len(actions)))

            for i, action in enumerate(actions):
                action_index = action[1]
                child_id = leaf_id + (action_index,)
                child_board = utils.get_board(child_id, self.board_size)
                next_turn = utils.get_turn(child_id)

                prior_p = prior_prob[action_index]

                if self.noise:
                    if leaf_id == self.root_id:
                        prior_p = 0.75 * prior_p + 0.25 * noise_probs[i]
                        # print("add noise expansion")

                self.tree[child_id] = {'board': child_board,
                                       'player': next_turn,
                                       'parent': leaf_id,
                                       'child': [],
                                       'n': 0.,
                                       'w': 0.,
                                       'q': 0.,
                                       'p': prior_p}

                self.tree[leaf_id]['child'].append(action_index)

            reward = False
            return value, reward
        else:
            # print("terminal node don't expansion")
            reward = utils.get_reward(win_index, leaf_id)
            value = False
            return value, reward

    def _backup(self, leaf_id, value, reward):
        # print("backup")
        node_id = leaf_id
        count = 0

        while node_id != self.tree[self.root_id]['parent']:
            # print('id: {}, reward: {}'.format(node_id,
            #                                   reward * (-1)**(count)))
            self.tree[node_id]['n'] += 1

            if not reward:
                self.tree[node_id]['w'] += (-value) * (-1)**(count)
                count += 1
                # print('value:', -value)
            else:
                self.tree[node_id]['w'] += reward * (-1)**(count)
                # print('reward:', reward * (-1)**(count))
                count += 1

            self.tree[node_id]['q'] = (self.tree[node_id]['w'] /
                                       self.tree[node_id]['n'])
            parent_id = self.tree[node_id]['parent']
            node_id = parent_id


class PUCTAgent(object):

    def __init__(self, board_size, num_mcts):
        self.board_size = board_size
        self.num_mcts = num_mcts
        # tictactoe and omok
        self.win_mark = 3 if board_size == 3 else 5
        self.alpha = 10 / self.board_size**2
        self.c_puct = 5
        self.root_id = None
        self.board = None
        self.turn = None
        self.tree = {}

    def reset(self):
        self.root_id = None
        self.turn = None
        self.tree.clear()

    def get_pi(self, root_id, board, turn, tau):
        self._init_mcts(root_id, board, turn)
        self._mcts(self.root_id)

        visit = np.zeros(self.board_size**2, 'float')

        for action_index in self.tree[self.root_id]['child']:
            child_id = self.root_id + (action_index,)
            visit[action_index] = self.tree[child_id]['n']

        if visit.max() > 1000:
            tau = 0.1

        pi = visit**(1 / tau)
        pi /= pi.sum()
        return pi

    def _init_mcts(self, root_id, board, turn):
        self.root_id = root_id
        self.board = board
        self.turn = turn

        if self.root_id not in self.tree:
            # print('init root node')
            self.tree[self.root_id] = {'board': self.board,
                                       'player': self.turn,
                                       'parent': None,
                                       'child': [],
                                       'n': 0.,
                                       'w': 0.,
                                       'q': 0.,
                                       'p': 0.}

    def _mcts(self, root_id):
        start = time.time()

        for i in range(self.num_mcts):
            sys.stdout.write('simulation: {}\r'.format(i + 1))
            sys.stdout.flush()
            leaf_id, win_index = self._selection(root_id)
            reward = self._expansion_simulation(leaf_id, win_index)
            self._backup(leaf_id, reward)

        finish = time.time() - start
        print("{} simulations end ({:0.0f}s)".format(i + 1, finish))

    def _selection(self, root_id):
        node_id = root_id

        while self.tree[node_id]['n'] > 0:
            win_index = utils.check_win(
                self.tree[node_id]['board'], self.win_mark)

            if win_index != 0:
                return node_id, win_index

            qu = {}
            ids = []
            total_n = 0

            for action_idx in self.tree[node_id]['child']:
                edge_id = node_id + (action_idx,)
                n = self.tree[edge_id]['n']
                total_n += n

            for action_index in self.tree[node_id]['child']:
                # total_n = self.tree[node_id]['n']
                child_id = node_id + (action_index,)
                n = self.tree[child_id]['n']
                q = self.tree[child_id]['q']
                p = self.tree[child_id]['p']
                u = self.c_puct * p * np.sqrt(total_n) / (n + 1)
                qu[child_id] = q + u

            max_value = max(qu.values())
            ids = [key for key, value in qu.items()
                   if value == max_value]
            node_id = ids[np.random.choice(len(ids))]
            # print('selectionted id:', node_id)

        win_index = utils.check_win(self.tree[node_id]['board'],
                                    self.win_mark)
        return node_id, win_index

    def _expansion_simulation(self, leaf_id, win_index):
        leaf_board = self.tree[leaf_id]['board']
        current_player = self.tree[leaf_id]['player']

        if win_index == 0:
            # print("expansion")
            actions = utils.legal_actions(leaf_board)
            prior_prob = 1 / len(actions)

            for i, action in enumerate(actions):
                action_index = action[1]
                child_id = leaf_id + (action_index,)
                child_board = utils.get_board(child_id, self.board_size)
                next_turn = utils.get_turn(child_id)

                self.tree[child_id] = {'board': child_board,
                                       'player': next_turn,
                                       'parent': leaf_id,
                                       'child': [],
                                       'n': 0.,
                                       'w': 0.,
                                       'q': 0.,
                                       'p': prior_prob}

                self.tree[leaf_id]['child'].append(action_index)

            if self.tree[leaf_id]['parent']:
                # print("simulation")
                board_sim = leaf_board.copy()
                turn_sim = current_player

                while True:
                    actions_sim = utils.legal_actions(board_sim)
                    action_sim = actions_sim[
                        np.random.choice(len(actions_sim))]
                    coord_sim = action_sim[0]

                    if turn_sim == 0:
                        board_sim[coord_sim] = 1
                    else:
                        board_sim[coord_sim] = -1

                    win_idx_sim = utils.check_win(board_sim, self.win_mark)

                    if win_idx_sim == 0:
                        turn_sim = abs(turn_sim - 1)

                    else:
                        reward = utils.get_reward(win_idx_sim, leaf_id)
                        return reward
            else:
                # print("root node don't simulation")
                reward = 0.
                return reward
        else:
            # print("terminal node don't expansion")
            reward = utils.get_reward(win_index, leaf_id)
            return reward

    def _backup(self, leaf_id, reward):
        # print("backup")
        node_id = leaf_id
        count = 0

        while node_id != self.tree[self.root_id]['parent']:
            # print('id: {}, reward: {}'.format(node_id,
            #                                   reward * (-1)**(count)))
            self.tree[node_id]['n'] += 1
            self.tree[node_id]['w'] += reward * (-1)**(count)
            self.tree[node_id]['q'] = (self.tree[node_id]['w'] /
                                       self.tree[node_id]['n'])
            parent_id = self.tree[node_id]['parent']
            node_id = parent_id
            count += 1


class UCTAgent(object):

    def __init__(self, board_size, num_mcts):
        self.board_size = board_size
        self.num_mcts = num_mcts
        # tictactoe and omok
        self.win_mark = 3 if board_size == 3 else 5
        self.root_id = None
        self.board = None
        self.turn = None
        self.tree = {}

    def reset(self):
        self.root_id = None
        self.board = None
        self.turn = None
        self.tree.clear()

    def get_pi(self, root_id, board, turn, tau):
        self._init_mcts(root_id, board, turn)
        self._mcts(self.root_id)

        root_node = self.tree[self.root_id]
        q = np.ones(self.board_size**2, 'float') * -np.inf
        pi = np.zeros(self.board_size**2, 'float')

        for action_index in root_node['child']:
            child_id = self.root_id + (action_index,)
            q[action_index] = self.tree[child_id]['q']

        max_idx = np.argwhere(q == q.max())
        pi[max_idx[np.random.choice(len(max_idx))]] = 1
        return pi

    def _init_mcts(self, root_id, board, turn):
        self.root_id = root_id
        self.board = board
        self.turn = turn

        if self.root_id not in self.tree:
            # print('init root node')
            self.tree[self.root_id] = {'board': self.board,
                                       'player': self.turn,
                                       'parent': None,
                                       'child': [],
                                       'n': 0.,
                                       'w': 0.,
                                       'q': 0.}

    def _mcts(self, root_id):
        start = time.time()

        for i in range(self.num_mcts):
            sys.stdout.write('simulation: {}\r'.format(i + 1))
            sys.stdout.flush()
            leaf_id, win_index = self._selection(root_id)
            reward = self._expansion_simulation(leaf_id, win_index)
            self._backup(leaf_id, reward)

        finish = time.time() - start
        print("{} simulations end ({:0.0f}s)".format(i + 1, finish))

    def _selection(self, root_id):
        node_id = root_id

        while self.tree[node_id]['n'] > 0:
            win_index = utils.check_win(
                self.tree[node_id]['board'], self.win_mark)

            if win_index != 0:
                return node_id, win_index

            qu = {}
            ids = []
            total_n = 0

            for action_index in self.tree[node_id]['child']:
                child_id = node_id + (action_index,)
                n = self.tree[child_id]['n']
                q = self.tree[child_id]['q']
                total_n = self.tree[node_id]['n']

                if n == 0:
                    u = np.inf
                else:
                    u = np.sqrt(2 * np.log(total_n) / n)

                qu[child_id] = q + u

            max_value = max(qu.values())
            ids = [key for key, value in qu.items()
                   if value == max_value]
            node_id = ids[np.random.choice(len(ids))]
            # print('selectionted id:', node_id)

        win_index = utils.check_win(self.tree[node_id]['board'],
                                    self.win_mark)
        return node_id, win_index

    def _expansion_simulation(self, leaf_id, win_index):
        leaf_board = self.tree[leaf_id]['board']
        current_player = self.tree[leaf_id]['player']

        if win_index == 0:
            # print("expansion")
            actions = utils.legal_actions(leaf_board)

            for action in actions:
                action_index = action[1]
                child_id = leaf_id + (action_index,)
                child_board = utils.get_board(child_id, self.board_size)
                next_turn = utils.get_turn(child_id)

                self.tree[child_id] = {'board': child_board,
                                       'player': next_turn,
                                       'parent': leaf_id,
                                       'child': [],
                                       'n': 0.,
                                       'w': 0.,
                                       'q': 0.}

                self.tree[leaf_id]['child'].append(action_index)

            if self.tree[leaf_id]['parent']:
                # print("simulation")
                board_sim = leaf_board.copy()
                turn_sim = current_player

                while True:
                    actions_sim = utils.legal_actions(board_sim)
                    action_sim = actions_sim[
                        np.random.choice(len(actions_sim))]
                    coord_sim = action_sim[0]

                    if turn_sim == 0:
                        board_sim[coord_sim] = 1
                    else:
                        board_sim[coord_sim] = -1

                    win_idx_sim = utils.check_win(board_sim, self.win_mark)

                    if win_idx_sim == 0:
                        turn_sim = abs(turn_sim - 1)

                    else:
                        reward = utils.get_reward(win_idx_sim, leaf_id)
                        return reward
            else:
                # print("root node don't simulation")
                reward = 0.
                return reward
        else:
            # print("terminal node don't expansion")
            reward = utils.get_reward(win_index, leaf_id)
            return reward

    def _backup(self, leaf_id, reward):
        # print("backup")
        node_id = leaf_id
        count = 0

        while node_id != self.tree[self.root_id]['parent']:
            # print('id: {}, reward: {}'.format(node_id,
            #                                   reward * (-1)**(count)))
            self.tree[node_id]['n'] += 1
            self.tree[node_id]['w'] += reward * (-1)**(count)
            self.tree[node_id]['q'] = (self.tree[node_id]['w'] /
                                       self.tree[node_id]['n'])
            parent_id = self.tree[node_id]['parent']
            node_id = parent_id
            count += 1


class RandomAgent(object):

    def __init__(self, board_size):
        self.board_size = board_size

    def get_pi(self, root_id, board, turn, tau):
        self.root_id = root_id
        action = utils.legal_actions(board)
        prob = 1 / len(action)
        pi = np.zeros(self.board_size**2, 'float')

        for loc, idx in action:
            pi[idx] = prob

        return pi

    def reset(self):
        self.root_id = None


class HumanAgent(object):

    COLUMN = {"a": 0, "b": 1, "c": 2,
              "d": 3, "e": 4, "f": 5,
              "g": 6, "h": 7, "i": 8,
              "j": 9, "k": 10, "l": 11,
              "m": 12, "n": 13, "o": 14}

    def __init__(self, board_size):
        self.board_size = board_size
        self._init_board_label()

    def get_pi(self, root_id, board, turn, tau):
        self.root_id = root_id

        while True:
            try:
                action_index = self.input_action(self.last_label)
            except Exception:
                continue
            else:
                pi = np.zeros(self.board_size**2, 'float')
                pi[action_index] = 1
                return pi

    def _init_board_label(self):
        self.last_label = str(self.board_size)

        for k, v in self.COLUMN.items():
            if v == self.board_size - 1:
                self.last_label += k
                break

    def input_action(self, last_label):
        action_coord = input('1a ~ {}: '.format(last_label)).rstrip().lower()
        row = int(action_coord[0]) - 1
        col = self.COLUMN[action_coord[1]]
        action_index = row * self.board_size + col
        return action_index

    def reset(self):
        self.root_id = None
