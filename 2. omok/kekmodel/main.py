'''
Author : Woonwon Lee, Jungdae Kim
Data : 2018.03.12, 2018.03.28
Project : Make your own Alpha Zero
'''
from utils import *
# from model import AlphaZero
from neural_net import PVNet
# from copy import deepcopy
# import time
import sys
import numpy as np
import random
from collections import deque
import torch
import torch.optim as optim
from torch.autograd import Variable
import pygame
import env_small as game
from agent import Player

N_BLOCKS = 10
IN_PLANES = 17
OUT_PLANES = 128
BATCH_SIZE = 4
LR = 0.01
L2 = 0.0001

STATE_SIZE = 9
NUM_MCTS = 200


def self_play(num_episode):
    tau_thres = 6
    # Game Loop
    for episode in range(num_episode):
        print('playing ', episode + 1, 'th episode by self-play')
        env = game.GameState('text')
        game_board = np.zeros([STATE_SIZE, STATE_SIZE])
        samples_black = []
        samples_white = []
        turn = 0
        win_index = 0
        step = 0

        while win_index == 0:
            render_str(game_board, STATE_SIZE)
            pi = agent.get_pi(game_board, turn)
            print('')
            print(pi.reshape(STATE_SIZE, STATE_SIZE).round(decimals=4))
            state = get_state_pt(agent.root_id, turn, STATE_SIZE, IN_PLANES)

            if step < tau_thres:
                tau = 1
            else:
                tau = 0
            # ====================== get_action ======================
            action, action_index = get_action(pi, tau)
            agent.root_id += (action_index,)
            if turn == 0:
                samples_black.append([state, pi])
            else:
                samples_white.append([state, pi])

            game_board, _, check_valid_pos, win_index, turn, _ = env.step(action)
            step += 1

            if not check_valid_pos:
                raise ValueError("no legal move!")

            if win_index != 0:
                render_str(game_board, STATE_SIZE)
                print("win is ", win_index, "in episode", episode + 1)
                agent.reset()

                if win_index == 1:
                    reward_black = 1
                    reward_white = -1
                elif win_index == 2:
                    reward_black = -1
                    reward_white = 1
                else:
                    reward_black = 0
                    reward_white = 0

                for i in range(len(samples_black)):
                    memory.append([samples_black[i][0], samples_black[i][1], reward_black])

                for i in range(len(samples_white)):
                    memory.append([samples_white[i][0], samples_white[i][1], reward_white])
                break


def train(num_iter):
    optimizer = optim.SGD(agent.model.parameters(), lr=LR, momentum=0.9, weight_decay=L2)
    criterion_p = torch.nn.CrossEntropyLoss()
    criterion_v = torch.nn.MSELoss()

    for i in range(num_iter):
        sys.stdout.write('{} th iteration\r'.format(i + 1))
        sys.stdout.flush()
        mini_batch = random.sample(memory, BATCH_SIZE)
        mini_batch = np.array(mini_batch).transpose()
        state = np.vstack(mini_batch[0])
        pi = np.vstack(mini_batch[1])
        z = list(mini_batch[2])

        # state_input = np.reshape(state, [BATCH_SIZE, IN_PLANES, STATE_SIZE, STATE_SIZE])
        state_input = Variable(torch.FloatTensor(state))
        pi = Variable(torch.FloatTensor(pi))
        z = Variable(torch.FloatTensor(z))

        policy, value = agent.model(state_input)
        # policies = torch.sum(policies.mul(actions), dim=1)

        loss = criterion_v(value, z) + criterion_p(policy, pi)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def compete():
    pass


if __name__ == '__main__':
    memory = deque(maxlen=BATCH_SIZE * 10)
    agent = Player(STATE_SIZE, NUM_MCTS)
    agent.model = PVNet(N_BLOCKS, IN_PLANES, OUT_PLANES, STATE_SIZE)
    for i in range(1000):
        print('-----------------------------------------')
        print(i + 1, 'th training process')
        print('-----------------------------------------')
        self_play(num_episode=2)
        train(num_iter=1)
        # compete()
