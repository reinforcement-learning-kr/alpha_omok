# -*- coding: utf-8 -*-
# Pure MCTS for text Omok env
from env.env_text import OmokEnv, OmokEnvSimul

import time
from collections import deque, defaultdict

import numpy as np
from numpy import random
from xxhash import xxh64

N, Q = 0, 1
CURRENT = 0
OPPONENT = 1
COLOR = 2
BLACK = 1
WHITE = 0
BOARD_SIZE = 9

SIMULATIONS = 1600
GAMES = 10000


class MCTS:
    def __init__(self, num_simul):
        self.env_simul = OmokEnvSimul()
        self.num_simul = num_simul
        self.tree = None
        self.root = None
        self.state = None
        self.board = None
        self.legal_move = None
        self.no_legal_move = None
        self.ucb = None

        # used for backup
        self.key_memory = None
        self.action_memory = None

        # init
        self._reset()
        self.reset_tree()

    def _reset(self):
        self.key_memory = deque(maxlen=BOARD_SIZE**2)
        self.action_memory = deque(maxlen=BOARD_SIZE**2)

    def reset_tree(self):
        self.tree = defaultdict(lambda: np.zeros((BOARD_SIZE**2, 2), 'float'))

    def get_action(self, state):
        self.root = state.copy()
        self._simulation(state)
        # init root board after simulatons
        self.board = self.root.reshape(3, BOARD_SIZE**2)
        board_fill = self.board[CURRENT] + self.board[OPPONENT]
        self.legal_move = np.argwhere(board_fill == 0).flatten()
        self.no_legal_move = np.argwhere(board_fill != 0).flatten()
        # root state's key
        root_key = xxh64(self.root.tostring()).hexdigest()
        # argmax Q
        action = self._selection(root_key, c_ucb=0)
        print(self.ucb.reshape(BOARD_SIZE, BOARD_SIZE).round(decimals=4))
        return action

    def _simulation(self, state):
        start = time.time()
        print('Computing Moves', end='', flush=True)
        for sim in range(SIMULATIONS):
            if (sim + 1) % (160) == 0:
                print('.', end='', flush=True)
            # reset state
            self.state = self.env_simul.reset(state)
            done = False
            n_selection = 0
            n_expansion = 0
            while not done:
                # init board
                self.board = self.state.reshape(3, BOARD_SIZE**2)
                board_fill = self.board[CURRENT] + self.board[OPPONENT]
                self.legal_move = np.argwhere(board_fill == 0).flatten()
                self.no_legal_move = np.argwhere(board_fill != 0).flatten()
                key = xxh64(self.state.tostring()).hexdigest()
                # search my tree
                if key in self.tree:
                    # selection
                    action = self._selection(key, c_ucb=1)
                    self.action_memory.appendleft(action)
                    self.key_memory.appendleft(key)
                    n_selection += 1
                else:
                    if n_expansion == 0:
                        # expansion
                        action = self._expansion(key)
                        self.action_memory.appendleft(action)
                        self.key_memory.appendleft(key)
                        n_expansion += 1
                    else:
                        # rollout
                        action = random.choice(self.legal_move)
                self.state, reward, done = self.env_simul.step(action)
            if done:
                # backup & reset memory
                self._backup(reward, n_selection + n_expansion)
                self._reset()
        finish = round(time.time() - start)
        print('\n"{} Simulations End in {}s"'.format(sim + 1, finish))

    def _selection(self, key, c_ucb):
        edges = self.tree[key]
        # get ucb
        ucb = self._ucb(edges, c_ucb)
        self.ucb = ucb
        if self.board[COLOR][0] == WHITE:
            # black's choice
            action = np.argwhere(ucb == ucb.max()).flatten()
        else:
            # white's choice
            action = np.argwhere(ucb == ucb.min()).flatten()
        action = action[random.choice(len(action))]
        return action

    def _expansion(self, key):
        # only select once for rollout
        action = self._selection(key, c_ucb=1)
        return action

    def _ucb(self, edges, c_ucb):
        total_N = 0
        ucb = np.zeros((BOARD_SIZE**2), 'float')
        for i in range(BOARD_SIZE**2):
            total_N += edges[i][N]
        # black's ucb
        if self.board[COLOR][0] == WHITE:
            for move in self.legal_move:
                if edges[move][N] != 0:
                    ucb[move] = edges[move][Q] + c_ucb * \
                        np.sqrt(2 * np.log(total_N) / edges[move][N])
                else:
                    ucb[move] = np.inf
            for move in self.no_legal_move:
                ucb[move] = -np.inf
        # white's ucb
        else:
            for move in self.legal_move:
                if edges[move][N] != 0:
                    ucb[move] = edges[move][Q] - c_ucb * \
                        np.sqrt(2 * np.log(total_N) / edges[move][N])
                else:
                    ucb[move] = -np.inf
            for move in self.no_legal_move:
                ucb[move] = np.inf
        return ucb

    def _backup(self, reward, steps):
        # steps = n_selection + n_expansion
        # update edges in my tree
        for i in range(steps):
            edges = self.tree[self.key_memory[i]]
            action = self.action_memory[i]
            edges[action][N] += 1
            edges[action][Q] += (reward - edges[action][Q]) / edges[action][N]


def main():
    env = OmokEnv()
    mcts = MCTS(SIMULATIONS)
    result = {'Black': 0, 'White': 0, 'Draw': 0}
    for game in range(GAMES):
        print('#########  GAME: {}  #########\n'.format(game + 1))
        # reset state
        state = env.reset()
        done = False
        while not done:
            env.render()
            # start simulations
            action = mcts.get_action(state)
            state, z, done = env.step(action)
        if done:
            if z == 1:
                result['Black'] += 1
            elif z == -1:
                result['White'] += 1
            else:
                result['Draw'] += 1
            # render & reset tree
            env.render()
            mcts.reset_tree()
        # result
        print('')
        print("=" * 20, " {}  Game End  ".format(game + 1), "=" * 20)
        stat_game = ('Black Win: {}  White Win: {}  Draw: {}  Winrate: {:0.1f}%'.format(
            result['Black'], result['White'], result['Draw'],
            1 / (1 + np.exp(result['White'] / (game + 1)) /
                 np.exp(result['Black'] / (game + 1))) * 100))
        print(stat_game, '\n')


if __name__ == '__main__':
    main()
