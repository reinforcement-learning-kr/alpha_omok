# -*- coding: utf-8 -*-
import tictactoe_env
import tictactoe_env_simul
import neural_net_5block

import time
from collections import deque, defaultdict

import torch
from torch.autograd import Variable

import xxhash
import numpy as np

PLAYER, OPPONENT = 0, 1
MARK_O, MARK_X = 0, 1
N, W, Q, P = 0, 1, 2, 3
PLANE = np.zeros((3, 3), 'int').flatten()

GAMES = 20


class MCTS:

    def __init__(self, model_path=None, num_simul=800, num_channel=128, user=None):
        # simul env
        self.env_simul = tictactoe_env_simul.TicTacToeEnv()

        # tree
        self.tree = defaultdict(lambda: np.zeros((3, 3, 4), 'float'))

        # model
        self.pv_net = neural_net_5block.PolicyValueNet(num_channel).cuda()
        if model_path is not None:
            print(' #######  Model is loaded  ####### ')
            self.pv_net.load_state_dict(torch.load(model_path))

        self.done = False
        self.root = None
        self.evaluate = None
        self.player_color = None
        self.num_simul = num_simul
        self.user = user

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
        self.current_user = None

        # reset_episode member
        self.node_memory = None
        self.edge_memory = None
        self.action_memory = None
        self.action_count = None

        # init
        self.reset_step()
        self._reset_episode()

    def reset_step(self, current_user=None):
        self.edge = np.zeros((3, 3, 4), 'float')
        self.total_visit = 0
        self.legal_move = None
        self.no_legal_move = None
        self.state = None
        self.prob = np.zeros((3, 3), 'float')
        self.value = None
        self.current_user = current_user

    def _reset_episode(self):
        self.node_memory = deque(maxlen=9)
        self.edge_memory = deque(maxlen=9)
        self.action_memory = deque(maxlen=9)
        self.action_count = 0

    def select_action(self, state):
        if self.current_user is None:
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
        action = np.r_[self.current_user, move_target]
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
        state_variable = Variable(state_tensor.view(9, 3, 3).unsqueeze(0)).cuda()
        p_theta, v_theta = self.pv_net(state_variable)
        self.prob = p_theta.data.cpu().numpy()[0].reshape(3, 3)
        self.value = v_theta.data.cpu().numpy()[0]
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
        print("computing move...")
        for s in range(self.num_simul):
            state = self.env_simul.reset(root.copy(), self.player_color)
            done = False
            step = 0
            while not done:
                current_user = (self.user + step) % 2
                self.reset_step(current_user)
                action = self.select_action(state)
                state, reward, done_env, _ = self.env_simul.step(action)
                done = self.done or done_env
                step += 1
            if done:
                if self.done:
                    self.backup(self.value)
                else:
                    self.backup(reward)
        print('{} simulations end'.format(s + 1))
        self.current_user = self.user
        action = self.play(0)

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
        action = np.r_[self.current_user, final_move]
        print('v: ', self.evaluate.round(decimals=2))
        print('======   Pi   ======\n', pi.round(decimals=2), '\n')

        return tuple(action)


class AiVsAi:
    def __init__(self):
        self.ai_player = MCTS('data/model_s800_g800_e64xde.pickle', 800, 128, PLAYER)
        self.ai_oppoenet = MCTS(None, 800, 128, OPPONENT)
        self.current_user = None

    def select_action(self, state):
        if self.current_user == PLAYER:
            self.ai_player.current_user = PLAYER
            action = self.ai_player.simulation(state)
        else:
            self.ai_oppoenet.current_user = OPPONENT
            action = self.ai_oppoenet.simulation(state)
        return action


if __name__ == '__main__':
    env = tictactoe_env.TicTacToeEnv()
    manager = AiVsAi()
    result = {-1: 0, 0: 0, 1: 0}

    for game in range(GAMES):
        print('##########    Game: {}    ##########\n'.format(game + 1))
        player_color = (MARK_O + game) % 2
        state = env.reset(player_color=player_color)
        done = False
        step_play = 0

        while not done:
            current_user = ((PLAYER if player_color == MARK_O else OPPONENT) + step_play) % 2
            print('- BOARD -')
            print(env.board[PLAYER] + env.board[OPPONENT] * 2)
            manager.ai_player.player_color = player_color
            manager.ai_oppoenet.player_color = player_color
            if step_play < 2:
                manager.ai_player.tau = 1
                manager.ai_oppoenet.tau = 1
            else:
                manager.ai_player.tau = 0
                manager.ai_oppoenet.tau = 0
            manager.current_user = current_user
            action = manager.select_action(state)
            state, reward, done, _ = env.step(action)
            step_play += 1
        if done:
            result[reward] += 1
            print('- FINAL -')
            print(env.board[PLAYER] + env.board[OPPONENT] * 2, '\n')
            manager.ai_player = MCTS('data/model_s800_g800_e64xde.pickle', 800, 128, PLAYER)
            manager.ai_oppoenet = MCTS(None, 800, 128, OPPONENT)
            time.sleep(2)

    print('=' * 20, '\nWin: {}  Lose: {}  Draw: {}  Winrate: {:0.1f}%'.format(
        result[1], result[-1], result[0],
        1 / (1 + np.exp(result[-1] / GAMES) / np.exp(result[1] / GAMES)) * 100))
