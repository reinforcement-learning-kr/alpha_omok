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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

import env_small as game


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(24, action_size),
            nn.Softmax()
        )
        self.critic = nn.Sequential(
            nn.Linear(24, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        action_prob = self.actor(x)
        value = self.critic(x)
        return action_prob, value


class AlphaZero():
    def __init__(self, state_size, action_size):
        self.model = ActorCritic(state_size, action_size)

    def mcts(self):
        def selection(tree):
            return leaf_state

        def expansion(tree, leaf_state):
            action_probs, value = self.model.forward(leaf_state)
            # initialize new edges with action probs
            return tree, value

        def backup(tree, leaf_id, value):
            return tree


if __name__ == '__main__':
    env = game.GameState()
    state_size, win_mark = game.Return_BoardParams()

