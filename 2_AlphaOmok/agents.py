import sys
import time

import numpy as np
import torch

import utils

import threading

PRINT_MCTS = True
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

class Agent(object):
    def __init__(self, board_size):

        self.policy = np.zeros(board_size**2, 'float')
        self.visit = np.zeros(board_size**2, 'float')
        self.message = 'Hello'

    def get_policy(self):
        return self.policy        

    def get_visit(self):
        return self.visit  

    def get_name(self):
        return  type(self).__name__

    def get_message(self):
        return self.message

    def get_pv(self, root_id):
        return None, None      

class ZeroAgent(Agent):
    def __init__(self, board_size, num_mcts, inplanes, noise=True):
        super(ZeroAgent, self).__init__(board_size)
        self.board_size = board_size
        self.num_mcts = num_mcts
        self.inplanes = inplanes
        # tictactoe and omok
        self.win_mark = 3 if board_size == 3 else 5
        self.alpha = 10 / self.board_size**2
        self.c_puct = 5
        self.noise = noise
        self.root_id = None
        self.model = None
        self.tree = {}
        self.is_real_root = True

    def reset(self):
        self.root_id = None
        self.tree.clear()
        self.is_real_root = True

    def get_pi(self, root_id, tau):
        self._init_mcts(root_id)
        self._mcts(self.root_id)

        visit = np.zeros(self.board_size**2, 'float')
        policy = np.zeros(self.board_size**2, 'float')

        for action_index in self.tree[self.root_id]['child']:
            child_id = self.root_id + (action_index,)
            visit[action_index] = self.tree[child_id]['n']
            policy[action_index] = self.tree[child_id]['p']

        self.visit = visit
        self.policy = policy

        pi = visit / visit.sum()

        if tau == 0:
            pi, _ = utils.argmax_onehot(pi)

        return pi

    def _init_mcts(self, root_id):
        self.root_id = root_id
        if self.root_id not in self.tree:
            self.is_real_root = True
            # init root node
            self.tree[self.root_id] = {'child': [],
                                       'n': 0.,
                                       'w': 0.,
                                       'q': 0.,
                                       'p': 0.}
        # add noise
        else:
            self.is_real_root = False
            if self.noise:
                children = self.tree[self.root_id]['child']
                noise_probs = np.random.dirichlet(
                    self.alpha * np.ones(len(children)))

                for i, action_index in enumerate(children):
                    child_id = self.root_id + (action_index,)
                    self.tree[child_id]['p'] = 0.75 * \
                        self.tree[child_id]['p'] + 0.25 * noise_probs[i]

    def _mcts(self, root_id):
        start = time.time()
        if self.is_real_root:
            # do not count first expansion of the root node
            num_mcts = self.num_mcts + 1
        else:
            num_mcts = self.num_mcts

        for i in range(num_mcts):

            if PRINT_MCTS:
                sys.stdout.write('simulation: {}\r'.format(i + 1))
                sys.stdout.flush()

            self.message = 'simulation: {}\r'.format(i + 1)            

            # selection
            leaf_id, win_index = self._selection(root_id)

            # expansion and evaluation
            value, reward = self._expansion_evaluation(leaf_id, win_index)

            # backup
            self._backup(leaf_id, value, reward)

        finish = time.time() - start
        if PRINT_MCTS:
            print("{} simulations end ({:0.0f}s)".format(i + 1, finish))

    def _selection(self, root_id):
        node_id = root_id

        while self.tree[node_id]['n'] > 0:
            board = utils.get_board(node_id, self.board_size)
            win_index = utils.check_win(board, self.win_mark)

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
                child_id = node_id + (action_index,)
                n = self.tree[child_id]['n']
                q = self.tree[child_id]['q']
                p = self.tree[child_id]['p']
                u = self.c_puct * p * np.sqrt(total_n) / (n + 1)
                qu[child_id] = q + u

            max_value = max(qu.values())
            ids = [key for key, value in qu.items() if value == max_value]
            node_id = ids[np.random.choice(len(ids))]

        board = utils.get_board(node_id, self.board_size)
        win_index = utils.check_win(board, self.win_mark)

        return node_id, win_index

    def _expansion_evaluation(self, leaf_id, win_index):
        leaf_state = utils.get_state_pt(
            leaf_id, self.board_size, self.inplanes)
        self.model.eval()
        with torch.no_grad():
            state_input = torch.tensor([leaf_state]).to(device).float()
            policy, value = self.model(state_input)
            policy = policy.cpu().numpy()[0]
            value = value.cpu().numpy()[0]

        if win_index == 0:
            # expansion
            actions = utils.legal_actions(leaf_id, self.board_size)
            prior_prob = np.zeros(self.board_size**2)

            # re-nomalization
            for action_index in actions:
                prior_prob[action_index] = policy[action_index]

            prior_prob /= prior_prob.sum()

            if self.noise:
                # root node noise
                if leaf_id == self.root_id:
                    noise_probs = np.random.dirichlet(
                        self.alpha * np.ones(len(actions)))

            for i, action_index in enumerate(actions):
                child_id = leaf_id + (action_index,)

                prior_p = prior_prob[action_index]

                if self.noise:
                    if leaf_id == self.root_id:
                        prior_p = 0.75 * prior_p + 0.25 * noise_probs[i]

                self.tree[child_id] = {'child': [],
                                       'n': 0.,
                                       'w': 0.,
                                       'q': 0.,
                                       'p': prior_p}

                self.tree[leaf_id]['child'].append(action_index)
            # return value
            reward = False
            return value, reward
        else:
            # terminal node
            # return reward
            reward = 1.
            value = False
            return value, reward

    def _backup(self, leaf_id, value, reward):
        node_id = leaf_id
        count = 0
        while node_id != self.root_id[:-1]:
            self.tree[node_id]['n'] += 1

            if not reward:
                self.tree[node_id]['w'] += (-value) * (-1)**(count)
                count += 1
            else:
                self.tree[node_id]['w'] += reward * (-1)**(count)
                count += 1

            self.tree[node_id]['q'] = (self.tree[node_id]['w'] /
                                       self.tree[node_id]['n'])
            parent_id = node_id[:-1]
            node_id = parent_id

    def del_parents(self, root_id):
        max_len = 0
        if self.tree:
            for key in list(self.tree.keys()):
                if len(key) > max_len:
                    max_len = len(key)
                if len(key) < len(root_id):
                    del self.tree[key]
        print('tree size:', len(self.tree))
        print('tree depth:', 0 if max_len <= 0 else max_len - 1)

    def get_pv(self, root_id):
        state = utils.get_state_pt(root_id, self.board_size, self.inplanes)
        self.model.eval()
        with torch.no_grad():
            state_input = torch.tensor([state]).to(device).float()
            policy, value = self.model(state_input)
            p = policy.data.cpu().numpy()[0]
            v = value.data.cpu().numpy()[0]
        return p, v

class PUCTAgent(Agent):
    def __init__(self, board_size, num_mcts):
        super(PUCTAgent, self).__init__(board_size)    
        self.board_size = board_size
        self.num_mcts = num_mcts
        # tictactoe and omok
        self.win_mark = 3 if board_size == 3 else 5
        self.c_puct = 5
        self.root_id = None
        self.board = None
        self.turn = None
        self.tree = {}
        self.is_real_root = True

    def reset(self):
        self.is_real_root = True
        self.root_id = None
        self.turn = None
        self.tree.clear()

    def get_pi(self, root_id, board, turn, tau):
        self._init_mcts(root_id, board, turn)
        self._mcts(self.root_id)

        visit = np.zeros(self.board_size**2, 'float')
        pi = np.zeros(self.board_size**2, 'float')

        for action_index in self.tree[self.root_id]['child']:
            child_id = self.root_id + (action_index,)
            visit[action_index] = self.tree[child_id]['n']

        max_idx = np.argwhere(visit == visit.max())
        pi[max_idx[np.random.choice(len(max_idx))]] = 1

        return pi

    def _init_mcts(self, root_id, board, turn):
        self.root_id = root_id
        self.board = board
        self.turn = turn
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

        for i in range(self.num_mcts + 1):
            sys.stdout.write('simulation: {}\r'.format(i + 1))
            sys.stdout.flush()
            leaf_id, win_index = self._selection(root_id)
            reward = self._expansion_simulation(leaf_id, win_index)
            self._backup(leaf_id, reward)

        finish = time.time() - start
        print("{} simulations end ({:0.0f}s)".format(self.num_mcts, finish))

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

        win_index = utils.check_win(self.tree[node_id]['board'],
                                    self.win_mark)
        return node_id, win_index

    def _expansion_simulation(self, leaf_id, win_index):
        leaf_board = self.tree[leaf_id]['board']
        current_player = self.tree[leaf_id]['player']

        if win_index == 0:
            actions = utils.valid_actions(leaf_board)
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
                board_sim = leaf_board.copy()
                turn_sim = current_player

                while True:
                    actions_sim = utils.valid_actions(board_sim)
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
                # root node don't simulation
                reward = 0.
                return reward
        else:
            # terminal node don't expansion
            reward = 1.
            return reward

    def _backup(self, leaf_id, reward):
        node_id = leaf_id
        count = 0

        while node_id is not None:
            self.tree[node_id]['n'] += 1
            self.tree[node_id]['w'] += reward * (-1)**(count)
            self.tree[node_id]['q'] = (self.tree[node_id]['w'] /
                                       self.tree[node_id]['n'])
            parent_id = self.tree[node_id]['parent']
            node_id = parent_id
            count += 1

    def del_parents(self, root_id):
        max_len = 0
        if self.tree:
            for key in list(self.tree.keys()):
                if len(key) > max_len:
                    max_len = len(key)
                if len(key) < len(root_id):
                    del self.tree[key]
        print('tree size:', len(self.tree))
        print('tree depth:', 0 if max_len <= 0 else max_len - 1)


class UCTAgent(Agent):
    def __init__(self, board_size, num_mcts):
        super(UCTAgent, self).__init__(board_size)   
        self.board_size = board_size
        self.num_mcts = num_mcts
        # tictactoe and omok
        self.win_mark = 3 if board_size == 3 else 5
        self.root_id = None
        self.board = None
        self.turn = None
        self.tree = {}
        self.is_real_root = True

    def reset(self):
        self.is_real_root = True
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
        self.is_real_root = True
        # init root node
        self.tree[self.root_id] = {'board': self.board,
                                   'player': self.turn,
                                   'parent': None,
                                   'child': [],
                                   'n': 0.,
                                   'w': 0.,
                                   'q': 0.}

    def _mcts(self, root_id):
        start = time.time()

        if self.is_real_root:
            num_mcts = self.num_mcts + 1
        else:
            num_mcts = self.num_mcts

        for i in range(num_mcts):

            mcts_count = i

            sys.stdout.write('simulation: {}\r'.format(i + 1))
            sys.stdout.flush()
            leaf_id, win_index = self._selection(root_id)
            reward = self._expansion_simulation(leaf_id, win_index)
            self._backup(leaf_id, reward)

        finish = time.time() - start
        print("{} simulations end ({:0.0f}s)".format(self.num_mcts, finish))

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
                child_id = node_id + (action_index,)
                n = self.tree[child_id]['n']
                q = self.tree[child_id]['q']

                if n == 0:
                    u = np.inf
                else:
                    u = np.sqrt(2 * np.log(total_n) / n)

                qu[child_id] = q + u

            max_value = max(qu.values())
            ids = [key for key, value in qu.items()
                   if value == max_value]
            node_id = ids[np.random.choice(len(ids))]

        win_index = utils.check_win(self.tree[node_id]['board'],
                                    self.win_mark)
        return node_id, win_index

    def _expansion_simulation(self, leaf_id, win_index):
        leaf_board = self.tree[leaf_id]['board']
        current_player = self.tree[leaf_id]['player']

        if win_index == 0:
            # expansion
            actions = utils.valid_actions(leaf_board)

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
                # simulation
                board_sim = leaf_board.copy()
                turn_sim = current_player

                while True:
                    actions_sim = utils.valid_actions(board_sim)
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
                # root node don't simulation
                reward = 0.
                return reward
        else:
            # terminal node don't expansion
            reward = 1.
            return reward

    def _backup(self, leaf_id, reward):

        node_id = leaf_id
        count = 0

        while node_id is not None:
            self.tree[node_id]['n'] += 1
            self.tree[node_id]['w'] += reward * (-1)**(count)
            self.tree[node_id]['q'] = (self.tree[node_id]['w'] /
                                       self.tree[node_id]['n'])
            parent_id = self.tree[node_id]['parent']
            node_id = parent_id
            count += 1

    def del_parents(self, root_id):
        max_len = 0
        if self.tree:
            for key in list(self.tree.keys()):
                if len(key) > max_len:
                    max_len = len(key)
                if len(key) < len(root_id):
                    del self.tree[key]
        print('tree size:', len(self.tree))
        print('tree depth:', 0 if max_len <= 0 else max_len - 1)


class RandomAgent(Agent):
    def __init__(self, board_size):
        super(RandomAgent, self).__init__(board_size)
        self.board_size = board_size

    def get_pi(self, root_id, board, turn, tau):
        self.root_id = root_id
        action = utils.valid_actions(board)
        prob = 1 / len(action)
        pi = np.zeros(self.board_size**2, 'float')

        for loc, idx in action:
            pi[idx] = prob

        return pi

    def reset(self):
        self.root_id = None

    def del_parents(self, root_id):
        return


class HumanAgent(Agent):
    COLUMN = {"a": 0, "b": 1, "c": 2,
              "d": 3, "e": 4, "f": 5,
              "g": 6, "h": 7, "i": 8,
              "j": 9, "k": 10, "l": 11,
              "m": 12, "n": 13, "o": 14}

    def __init__(self, board_size, env):
        super(HumanAgent, self).__init__(board_size)
        self.board_size = board_size
        self._init_board_label()
        self.root_id = (0,)
        self.env = env

    def get_pi(self, root_id, board, turn, tau):
        self.root_id = root_id

        while True:
            action = 0

            _, check_valid_pos, _, _, action_index = self.env.step(
                action)

            if check_valid_pos is True:
                pi = np.zeros(self.board_size**2, 'float')
                pi[action_index] = 1
                break

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

    def del_parents(self, root_id):
        return

class WebAgent(Agent):

    def __init__(self, board_size):
        super(WebAgent, self).__init__(board_size)
        self.board_size = board_size
        self.root_id = (0,)
        self.wait_action_idx = -1
        self.cv = threading.Condition()

    def get_pi(self, root_id, board, turn, tau):
        self.root_id = root_id

        self.cv.acquire()
        while self.wait_action_idx == -1:
            self.cv.wait()

        action_index = self.wait_action_idx
        self.wait_action_idx = -1

        self.cv.release()

        pi = np.zeros(self.board_size**2, 'float')
        pi[action_index] = 1

        return pi

    def put_action(self, action_idx):

        if action_idx < 0 and action_idx >= self.board_size**2:
            return

        self.cv.acquire()
        self.wait_action_idx = action_idx
        self.cv.notifyAll()
        self.cv.release()

    def reset(self):
        self.root_id = None

    def del_parents(self, root_id):
        return