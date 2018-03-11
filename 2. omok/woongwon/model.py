'''
author : Woonwon Lee
data : 2018.03.08
project : make your own alphazero
'''
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, action_size):
        super(ActorCritic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(2592, action_size),
            nn.Softmax()
        )
        self.critic = nn.Sequential(
            nn.Linear(2592, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        action_prob = self.actor(x)
        value = self.critic(x)
        return action_prob, value


class AlphaZero:
    def __init__(self, action_size):
        self.model = ActorCritic(action_size)

