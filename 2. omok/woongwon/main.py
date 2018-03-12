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
    game_board = np.zeros([state_size, state_size])

    # Game Loop
    for i in range(100):
        # Select action
        action = np.zeros(action_size)
        game_board = np.reshape(game_board, [1, 1, state_size, state_size])
        state = torch.from_numpy(game_board)
        state = Variable(state).float().cpu()
        policy, value = agent.model.model(state)
        policy = policy.data.numpy()[0]
        action_index = int(np.random.choice(action, p=policy))
        action[action_index] = 1

        # Find legal moves
        count_move = 0
        legal_index = []
        for i in range(state_size):
            for j in range(state_size):
                # Append legal move index into list
                if game_board[i, j] == 0:
                    legal_index.append(count_move)
                count_move += 1

        # Randomly take action among legal actions
        if len(legal_index) > 0:
            action[random.choice(legal_index)] = 1

        # Take action and get info. for update
        game_board, state, check_valid_pos, win_index, turn = env.step(action)


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

