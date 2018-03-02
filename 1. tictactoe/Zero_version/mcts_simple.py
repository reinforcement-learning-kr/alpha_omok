# -*- coding: utf-8 -*-
import tictactoe_env
import tictactoe_env_simul
import neural_net_5block

import time
import pickle
from collections import deque, defaultdict

import torch
from torch.autograd import Variable

import xxhash
import slackweb
import numpy as np

PLAYER, OPPONENT = 0, 1
MARK_O, MARK_X = 0, 1
N, W, Q, P = 0, 1, 2, 3
PLANE = np.zeros((3, 3), 'int').flatten()

GAMES = 1


class MCTS:

    def __init__(self, model_path=None, num_simul=800, num_channel=128):
        # simul env
        self.env_simul = tictactoe_env_simul.TicTacToeEnv()

        # tree
        self.tree = defaultdict(lambda: np.zeros((3, 3, 4), 'float'))

        # model
        self.pv_net = neural_net_5block.PolicyValueNet(num_channel)
        if model_path is not None:
            print(' #######  Model is loaded  #######')
            self.pv_net.load_state_dict(torch.load(model_path))

        self.done = False
        self.root = None
        self.evaluate = None
        self.player_color = None
        self.current_user_game = None
        self.current_user_simul = None
        self.num_simul = num_simul

        # hyperparameter
        self.c_puct = 5
        self.epsilon = 0.25
        self.alpha = 0.7
        self.tau = None

        # reset_step member
        self.edge = None
        self.total_visit = None
        self.legal_move = None
        self.no_legal_move = None
        self.state = None
        self.prob = None
        self.value = None
        self.simul_user = None

        # reset_episode member
        self.node_memory = None
        self.edge_memory = None
        self.action_memory = None
        self.action_count = None

        # init
        self.reset_step()
        self._reset_episode()

    def reset_step(self):
        self.edge = np.zeros((3, 3, 4), 'float')
        self.total_visit = 0
        self.legal_move = None
        self.no_legal_move = None
        self.state = None
        self.prob = np.zeros((3, 3), 'float')
        self.value = None

    def _reset_episode(self):
        self.node_memory = deque(maxlen=9)
        self.edge_memory = deque(maxlen=9)
        self.action_memory = deque(maxlen=9)
        self.action_count = 0

    def select_action(self, state):
        if self.current_user_simul is None:
            raise NotImplementedError("Set Current User!")

        self.action_count += 1

        self.state = state
        node = xxhash.xxh64(self.state.tostring()).hexdigest()
        self.node_memory.appendleft(node)

        origin_state = state.reshape(9, 3, 3)
        board_fill = origin_state[0] + origin_state[4]
        self.legal_move = np.argwhere(board_fill == 0)
        self.no_legal_move = np.argwhere(board_fill != 0)

        self._tree_search(node)
        puct = self._puct(self.edge)
        puct_max = np.argwhere(puct == puct.max())
        move_target = puct_max[np.random.choice(len(puct_max))]
        action = np.r_[self.current_user_simul, move_target]
        self.action_memory.appendleft(action)

        return tuple(action)

    def _tree_search(self, node):
        if node in self.tree:
            self.edge = self.tree[node]
            edge_n = np.zeros((3, 3), 'float')
            for i in range(3):
                for j in range(3):
                    self.prob[i, j] = self.edge[i, j][P]
                    edge_n[i, j] = self.edge[i, j][N]
            self.total_visit = np.sum(edge_n)
            self.done = False
        else:
            self._expand(node)

        if self.action_count == 1:
            for i, move in enumerate(self.legal_move):
                self.edge[tuple(move)][P] = (1 - self.epsilon) * self.prob[tuple(move)] + \
                    self.epsilon * np.random.dirichlet(
                        self.alpha * np.ones(len(self.legal_move)))[i]
        else:
            for move in self.legal_move:
                self.edge[tuple(move)][P] = self.prob[tuple(move)]

        self.edge_memory.appendleft(self.edge)

    def _puct(self, edge):
        puct = np.zeros((3, 3), 'float')
        for move in self.legal_move:
            puct[tuple(move)] = edge[tuple(move)][Q] + \
                self.c_puct * edge[tuple(move)][P] * \
                np.sqrt(self.total_visit) / (1 + edge[tuple(move)][N])
        for move in self.no_legal_move:
            puct[tuple(move)] = -np.inf

        return puct

    def _expand(self, node):
        self.edge = self.tree[node]
        state_tensor = torch.from_numpy(self.state).float()
        state_variable = Variable(state_tensor.view(9, 3, 3).unsqueeze(0))
        p_theta, v_theta = self.pv_net(state_variable)
        self.prob = p_theta.data.numpy()[0].reshape(3, 3)
        self.value = v_theta.data.numpy()[0]
        if np.array_equal(self.state, self.root):
            self.evaluate = self.value
        self.done = True

    def backup(self, reward):
        steps = self.action_count
        for i in range(steps):
            if self.action_memory[i][0] == PLAYER:
                self.edge_memory[i][tuple(
                    self.action_memory[i][1:])][
                    W] += reward
            else:
                self.edge_memory[i][tuple(
                    self.action_memory[i][1:])][
                    W] -= reward
            self.edge_memory[i][tuple(self.action_memory[i][1:])][N] += 1
            self.edge_memory[i][tuple(
                self.action_memory[i][1:])][Q] = self.edge_memory[i][tuple(
                    self.action_memory[i][1:])][W] / self.edge_memory[i][tuple(
                        self.action_memory[i][1:])][N]
            self.tree[self.node_memory[i]] = self.edge_memory[i]

        self._reset_episode()

    def simulation(self, root):
        self.root = root
        print("Computing Moves...")
        self.step_simul = 0
        for s in range(self.num_simul):
            state = self.env_simul.reset(root.copy(), self.player_color)
            done = False
            step = 0
            while not done:
                self.current_user_simul = (self.current_user_game + step) % 2
                action = self.select_action(state)
                state, reward, done_env, _ = self.env_simul.step(action)
                done = self.done or done_env
                step += 1
                self.step_simul += 1
            if done:
                if self.done:
                    self.backup(self.value)
                else:
                    self.backup(reward)
        print('"{} Simulations End!"'.format(s + 1))
        action = self.play(self.tau)
        return action

    def play(self, tau):
        root_node = xxhash.xxh64(self.root.tostring()).hexdigest()
        edge = self.tree[root_node]
        pi = np.zeros((3, 3), 'float')
        total_visit = 0
        action_space = []
        for i in range(3):
            for j in range(3):
                total_visit += edge[i, j][N]
                action_space.append([i, j])
        for i in range(3):
            for j in range(3):
                pi[i, j] = edge[i, j][N] / total_visit
        if tau == 0:
            deterministic = np.argwhere(pi == pi.max())
            final_move = deterministic[np.random.choice(len(deterministic))]
        else:
            stochactic = np.random.choice(9, p=pi.flatten())
            final_move = action_space[stochactic]
        action = np.r_[self.current_user_game, final_move]
        print(' `*`*  V: {}  `*`*'.format(self.evaluate.round(decimals=2)))
        print(' *=*=*=   Pi   =*=*=*\n', pi.round(decimals=2), '\n')

        state_memory.appendleft(self.root)
        pi_memory.appendleft(pi.flatten())
        return tuple(action)


if __name__ == '__main__':
    start = time.time()

    train_dataset_store = []
    state_memory = deque(maxlen=102400)
    pi_memory = deque(maxlen=102400)
    z_memory = deque(maxlen=102400)

    env = tictactoe_env.TicTacToeEnv()
    mcts = MCTS()
    result = {-1: 0, 0: 0, 1: 0}
    win_mark_o = 0
    step_game = 0
    step_simul = 0

    for game in range(GAMES):
        print('##########    Game: {}    ##########\n'.format(game + 1))
        player_color = (MARK_O + game) % 2
        state = env.reset(player_color=player_color)
        done = False
        step_play = 0

        while not done:
            print('---- BOARD ----')
            print(env.board[PLAYER] + env.board[OPPONENT] * 2.0)
            current_user = ((PLAYER if player_color == MARK_O else OPPONENT) + step_play) % 2
            mcts.player_color = player_color
            mcts.current_user_game = current_user
            if step_play < 2:
                mcts.tau = 1
            else:
                mcts.tau = 0
            action = mcts.simulation(state)
            state, z, done, _ = env.step(action)
            step_play += 1
            step_game += 1
        if done:
            print('==== FINAL ====')
            print(env.board[PLAYER] + env.board[OPPONENT] * 2.0, '\n')
            result[z] += 1
            for _ in range(step_play):
                z_memory.appendleft(z)
            step_simul += mcts.step_simul
            mcts = MCTS()
            if env.player_color == MARK_O:
                win_mark_o += 1

    train_dataset_store = list(zip(state_memory, pi_memory, z_memory))
    with open('data/train_dataset_s{}_g{}.pickle'.format(mcts.num_simul, game + 1), 'wb') as f:
        pickle.dump(train_dataset_store, f, pickle.HIGHEST_PROTOCOL)

    finish_game = round(float(time.time() - start))

    print("=" * 20, " {}  Game End  ".format(game + 1), "=" * 20)
    stat_game = ('Win: {}  Lose: {}  Draw: {}  Winrate: {:0.1f}%  WinMarkO: {}'.format(
        result[1], result[-1], result[0],
        1 / (1 + np.exp(result[-1] / (game + 1)) / np.exp(result[1] / (game + 1))) * 100,
        win_mark_o))
    print(stat_game)

    slack = slackweb.Slack(
        url="https://hooks.slack.com/services/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")
    slack.notify(
        text="Finished: [{} Game/{} Step] in {}s [Mac]".format(
            game + 1, step_game + step_simul, finish_game))
    slack.notify(text=stat_game)
