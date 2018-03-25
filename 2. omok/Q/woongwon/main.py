'''
author : Woonwon Lee
data : 2018.03.12
project : make your own alphazero
'''
from utils import valid_actions, check_win, render_str
from copy import deepcopy
import time
import torch
import random
import numpy as np
from collections import deque
import torch.optim as optim
from torch.autograd import Variable
import pygame
import env_small as game
from agent import Player

GAMEBOARD_SIZE = 9
memory = deque(maxlen=10000)
batch_size = 32


def append_sample(sample):
    memory.append(sample)


def self_play(num_episode):
    state = np.zeros([state_size, state_size, 17])
    game_board = np.zeros([state_size, state_size])
    # Game Loop
    for episode in range(num_episode):
        print('playing ', episode, 'th episode by self-play')
        samples_black = []
        samples_white = []
        turn = 0
        win_index = 0

        while win_index == 0:
            # render_str(game_board, GAMEBOARD_SIZE)
            # Select action

            state_input = np.reshape(state, [1, 17, state_size, state_size])
            state_input = torch.from_numpy(np.int32(state_input))
            state_input = Variable(state_input).float().cpu()

            policy = agent.get_policy(state_input, turn, agent.model.model)

            # policy, value = agent.model.model(state_input)
            # policy = policy.data.numpy()[0]

            # Find legal moves
            legal_policy = []
            legal_indexs = []
            for i in range(state_size):
                for j in range(state_size):
                    if game_board[i, j] == 0:
                        legal_indexs.append(state_size * i + j)
                        legal_policy.append(policy[state_size * i + j])

            legal_policy /= np.sum(legal_policy)
            if len(legal_policy) > 0:
                legal_index = np.random.choice(len(legal_policy), 1,
                                               p=legal_policy)[0]
                action_index = legal_indexs[legal_index]
                action = np.zeros(action_size)
                action[action_index] = 1

                if turn == 0:
                    samples_black.append([state, action])
                else:
                    samples_white.append([state, action])

                game_board, state, check_valid_pos, win_index, turn, _ \
                    = env.step(action)
            else:
                render_str(game_board, GAMEBOARD_SIZE)
                win_index = 3

            if win_index != 0:
                print("win is ", win_index, "in episode", episode)
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
                    memory.append([samples_black[i][0], samples_black[i][1],
                                   reward_black])

                for i in range(len(samples_white)):
                    memory.append([samples_white[i][0], samples_white[i][1],
                                   reward_white])
                break
        if (episode + 1) == num_episode:
            render_str(game_board, GAMEBOARD_SIZE)


def train(num_iter):
    optimizer = optim.Adam(agent.model.model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss(size_average=True)

    for i in range(num_iter):
        print(i, 'th iteration')
        mini_batch = random.sample(memory, batch_size)
        mini_batch = np.array(mini_batch).transpose()
        states = np.vstack(mini_batch[0])
        actions = np.vstack(mini_batch[1])
        rewards = list(mini_batch[2])

        states_input = np.reshape(states,
                                  [batch_size, 17, state_size, state_size])
        states_input = torch.Tensor(states_input)
        states_input = Variable(states_input).float()
        actions = Variable(torch.FloatTensor(actions))
        rewards = Variable(torch.FloatTensor(rewards))

        policies, values = agent.model.model(states_input)
        policies = torch.sum(policies.mul(actions), dim=1)

        loss = -criterion(policies, rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def compete():
    pass


if __name__ == '__main__':
    env = game.GameState('text')
    # win_mark == 5 : omok
    state_size, action_size, win_mark = game.Return_BoardParams()
    agent = Player(action_size)

    for i in range(1000):
        print('-----------------------------------------')
        print(i, 'th training process')
        self_play(num_episode=10)
        pygame.quit()
        train(num_iter=10)
        compete()
