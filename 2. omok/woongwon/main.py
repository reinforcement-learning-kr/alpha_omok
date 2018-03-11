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
from agent import Player


def self_play():
    pass


def train():
    pass


def compete():
    pass

if __name__ == '__main__':
    env = game.GameState()
    state_size, win_mark = game.Return_BoardParams()
    action_size = 81
    agent = Player(state_size, action_size)

    for i in range(1000):
        self_play()
        train()
        compete()

