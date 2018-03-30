'''
Author : Woonwon Lee, Jungdae Kim
Data : 2018.03.12, 2018.03.28
Project : Make your own Alpha Zero
'''
from utils import render_str, get_state_pt, get_action
# from model import AlphaZero
from neural_net import PVNet
import random
import numpy as np
from collections import deque, namedtuple
import torch
import torch.optim as optim
from torch.autograd import Variable
# import pygame
import env_small as game
from agent import Player

N_BLOCKS = 20
IN_PLANES = 17
OUT_PLANES = 128
BATCH_SIZE = 32
LR = 0.01
L2 = 0.0001

STATE_SIZE = 9
NUM_MCTS = 800


def self_play(num_episode):
    tau_thres = 6
    # Game Loop
    for episode in range(num_episode):
        print('playing ', episode + 1, 'th episode by self-play')
        env = game.GameState('text')
        board = np.zeros([STATE_SIZE, STATE_SIZE])
        samples_black = []
        samples_white = []
        turn = 0
        win_index = 0
        step = 0

        while win_index == 0:
            render_str(board, STATE_SIZE)
            pi = agent.get_pi(board, turn)
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
                samples_black.append((Tensor([state]), Tensor([pi])))
            else:
                samples_white.append((Tensor([state]), Tensor([pi])))

            board, _, check_valid_pos, win_index, turn, _ = env.step(action)
            step += 1

            if not check_valid_pos:
                raise ValueError("no legal move!")

            if win_index != 0:
                render_str(board, STATE_SIZE)
                print("win is ", win_index, "in episode", episode + 1)
                agent.reset()

                if win_index == 1:
                    reward_black = -1.
                    reward_white = 1.
                elif win_index == 2:
                    reward_black = 1.
                    reward_white = -1.
                else:
                    reward_black = 0.
                    reward_white = 0.

                for i in range(len(samples_black)):
                    memory.append(
                        NameTag(
                            samples_black[i][0],
                            samples_black[i][1],
                            Tensor([reward_black])
                        )
                    )

                for i in range(len(samples_white)):
                    memory.append(
                        NameTag(
                            samples_white[i][0],
                            samples_white[i][1],
                            Tensor([reward_white])
                        )
                    )
                break


def train(num_iter):
    optimizer = optim.SGD(
        agent.model.parameters(), lr=LR, momentum=0.9, weight_decay=L2)
    running_loss = 0.
    j = 0
    for i in range(num_iter):
        j += 1
        batch = random.sample(memory, BATCH_SIZE)
        batch = NameTag(*zip(*batch))

        s_batch = Variable(torch.cat(batch.s))
        pi_batch = Variable(torch.cat(batch.pi))
        z_batch = Variable(torch.cat(batch.z))

        p_batch, v_batch = agent.model(s_batch)

        pi_flat = pi_batch.view(1, BATCH_SIZE * STATE_SIZE**2)
        p_flat = p_batch.view(BATCH_SIZE * STATE_SIZE**2, 1)

        loss = (z_batch - v_batch).pow(2).sum() - \
            torch.matmul(pi_flat, torch.log(p_flat))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        print('{} iterarion loss: {:.4f}'.format(j, running_loss[0] / j))


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    print('cuda:', use_cuda)
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    NameTag = namedtuple('NameTag', ('s', 'pi', 'z'))
    memory = deque(maxlen=10000)
    agent = Player(STATE_SIZE, NUM_MCTS)
    agent.model = PVNet(N_BLOCKS, IN_PLANES, OUT_PLANES, STATE_SIZE)
    if use_cuda:
        agent.model.cuda()
    for i in range(100):
        print('-----------------------------------------')
        print(i + 1, 'th training process')
        print('-----------------------------------------')
        self_play(num_episode=3)
        train(num_iter=3)
        # compete()
