'''
author : Woonwon Lee
data : 2018.03.08
project : make your own alphazero
'''
from utils import valid_actions, check_win
from copy import deepcopy
import time
import sys
import gym
import torch
import pylab
import random
import numpy as np
from collections import deque
import torch.nn as nny
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

import env_small as game
from agent import Player


def self_play():
    state = np.zeros([state_size, state_size, 17])
    game_board = np.zeros([state_size, state_size])
    # Game Loop
    for i in range(100):
        # Select action
        state = np.reshape(state, [1, 17, state_size, state_size])
        state = torch.from_numpy(np.int32(state))
        state = Variable(state).float().cpu()
        policy, value = agent.model.model(state)
        policy = policy.data.numpy()[0]

        # Find legal moves
        legal_policy = []
        legal_indexs = []
        for i in range(state_size):
            for j in range(state_size):
                if game_board[i, j] == 0:
                    legal_indexs.append(state_size * i + j)
                    legal_policy.append(policy[state_size * i + j])

        legal_policy /= np.sum(legal_policy)
        legal_index = np.random.choice(len(legal_policy), 1, p=legal_policy)[0]
        action_index = legal_indexs[legal_index]
        action = np.zeros(action_size)
        action[action_index] = 1

        # Take action and get info. for update
        game_board, state, check_valid_pos, win_index, turn = env.step(action)
        time.sleep(0.2)


def train():
    pass


def compete():
    pass

if __name__ == '__main__':
    env = game.GameState()
    # win_mark == 5 : omok
    state_size, action_size, win_mark = game.Return_BoardParams()
    agent = Player(action_size)

    for i in range(1000):
        self_play()
        train()
        compete()

