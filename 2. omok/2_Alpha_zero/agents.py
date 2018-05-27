from copy import deepcopy
import sys
import time
#from pprint import pprint

import numpy as np
import torch
from torch.autograd import Variable

import utils


use_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


class Player(object):
    def __init__(self, board_size, num_mcts, inplanes):
        self.board_size = board_size
        self.num_mcts = num_mcts
        self.inplanes = inplanes
        self.win_mark = 3 if board_size == 3 else 5
        self.alpha = 10 / board_size**2
        self.turn = 0
        self.board = np.zeros([self.board_size, self.board_size])
        self.root_id = None
        self.model = None
        self.tree = {}
    
    def init_mcts(self, root_id, board, turn):
        self.root_id = root_id
        self.board = board
        self.turn = turn
    
    def selection(self, tree):
        node_id = self.root_id
        
        while True:
            if node_id in tree:
                num_child = len(tree[node_id]['child'])
            else:
                self.tree[node_id] = {'board': self.board,
                    'player': self.turn,
                        'child': [],
                            'parent': None,
                                'n': 0.,
                                    'w': 0.,
                                        'q': 0.,
                                            'p': 0.}
                num_child = 0
    
            # check if current node is leaf node
            if num_child == 0:
                return node_id

else:
    leaf_id = node_id
        
        if leaf_id == self.root_id:
            noise = np.random.dirichlet(
                                        self.alpha * np.ones(num_child))
                
                qu = {}
                
                for i in range(num_child):
                    action_index = tree[leaf_id]['child'][i]
                    child_id = leaf_id + (action_index,)
                    n = tree[child_id]['n']
                    q = tree[child_id]['q']
                    p = deepcopy(tree[child_id]['p'])
                    
                    if leaf_id == self.root_id:
                        p = 0.75 * p + 0.25 * noise[i]
                    
                    total_n = tree[tree[child_id]['parent']]['n'] - 1
                    u = 5. * p * np.sqrt(total_n) / (n + 1)
                    qu[child_id] = q + u
                
                max_value = max(qu.values())
                max_ids = []
                
                for key, value in qu.item():
                    if value == max_value:
                        max_ids.append(key)
        
                            node_id = max_ids[np.random.choice(len(max_ids))]

def expansion(self, tree, leaf_id):
    leaf_board = deepcopy(tree[leaf_id]['board'])
    win_index = utils.check_win(leaf_board, self.win_mark)
    
    if win_index == 0:
        actions = utils.legal_actions(leaf_board)
        current_player = tree[leaf_id]['player']
        leaf_state = utils.get_state_pt(
                                        leaf_id, current_player, self.board_size, self.inplanes)
            state_input = Variable(Tensor([leaf_state]))
            policy, value = self.model(state_input)
            policy = policy.data.cpu().numpy()[0]
            value = value.data.cpu().numpy()[0]
            
            for action in actions:
                board = deepcopy(tree[leaf_id]['board'])
                action_index = action[1]
                prior = policy[action_index]
                
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
                                                
                                                return tree, -value, None
                                        
                                            else:
                                                # If leaf node is terminal
                                                if win_index == 3:
                                                    reward = 0.
                                                        else:
                                                            reward = 1
                                                                
                                                                return tree, None, reward

def backup(self, tree, leaf_id, value, reward):
    node_id = leaf_id
        count = 0
        
        while True:
            if node_id == self.root_id:
                tree[node_id]['n'] += 1
                return tree
            
            tree[node_id]['n'] += 1
            
            if value:
                tree[node_id]['w'] += value
            
            elif reward:
                print('reward', reward * (-1)**(count))
                tree[node_id]['w'] += reward * (-1)**(count)
                count += 1
            
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
            self.tree, value, reward = self.expansion(self.tree, leaf_id)
            # step 3: backup
            self.tree = self.backup(self.tree, leaf_id, value, reward)
        
        finish = time.time() - start
        print("{} simulations end ({:0.0f}s)".format(i + 1, finish))
    
    def get_pi(self, root_id, board, turn):
        self.init_mcts(root_id, board, turn)
        self.mcts()
        root_node = self.tree[self.root_id]
        pi = np.zeros(self.board_size**2, 'float')
        
        for action_index in root_node['child']:
            child_id = self.root_id + (action_index,)
            pi[action_index] = self.tree[child_id]['n']
        
        # pi = np.exp(pi) / np.exp(pi).sum()
        pi /= pi.sum()
        return pi

def reset(self):
    self.turn = None
        self.board = None
        self.root_id = None
        self.tree.clear()


class PUCTAgent(object):
    
    C_PUCT = 5
    
    def __init__(self, board_size, num_mcts):
        self.board_size = board_size
        self.num_mcts = num_mcts
        # tictactoe and omok
        self.win_mark = 3 if board_size == 3 else 5
        self.alpha = 10 / self.board_size**2
        self.root_id = None
        self.board = None
        self.turn = None
        self.tree = {}
    
    def reset(self):
        self.root_id = None
        self.turn = None
        self.tree.clear()
    
    def get_pi(self, root_id, board, turn):
        self._init_mcts(root_id, board, turn)
        self._mcts(self.root_id)
        
        root_node = self.tree[self.root_id]
        pi = np.zeros(self.board_size**2, 'float')
        
        for action_index in root_node['child']:
            child_id = self.root_id + (action_index,)
            pi[action_index] = self.tree[child_id]['n']
        
        # pi = np.exp(pi) / np.exp(pi).sum()
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
                child_id = node_id + (action_index,)
                n = self.tree[child_id]['n']
                q = self.tree[child_id]['q']
                p = self.tree[child_id]['p']
                u = self.C_PUCT * p * np.sqrt(total_n) / (n + 1)
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
            # prior_probs = np.random.dirichlet(
            #     self.alpha * np.ones(len(actions)))
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

# pprint(self.tree)


class RandomAgent(object):
    def __init__(self, board_size):
        self.board_size = board_size
    
    def get_pi(self, root_id, board, turn):
        self.root_id = root_id
        action = utils.legal_actions(board)
        prob = 1 / len(action)
        pi = np.zeros(self.board_size**2, 'float')
        
        for loc, idx in action:
            pi[idx] = prob
        
        return pi
    
    def reset(self):
        pass


class HumanAgent(object):
    
    COLUMN = {"a": 0, "b": 1, "c": 2,
        "d": 3, "e": 4, "f": 5,
            "g": 6, "h": 7, "i": 8,
                "j": 9, "k": 10, "l": 11,
                    "m": 12, "n": 13, "o": 14}

def __init__(self, board_size):
    self.board_size = board_size
        self._init_board_label()

    def get_pi(self, root_id, board, turn):
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
        pass


if __name__ == '__main__':
    # test
    human = HumanAgent(9)
    human.get_pi(1, 1, 1)
