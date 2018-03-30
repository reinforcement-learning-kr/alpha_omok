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
from agent_test import Player

N_BLOCKS = 20
IN_PLANES = 5
OUT_PLANES = 128
BATCH_SIZE = 32
LR = 0.2
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
        samples = []
        turn = 0
        win_index = 0
        step = 0

        while win_index == 0:
            render_str(board, STATE_SIZE)
            # ====================  start mcts ======================
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
            samples.append((Tensor([state]), Tensor([pi])))
            board, check_valid_pos, win_index, turn, _ = env.step(action)
            step += 1

            if not check_valid_pos:
                raise ValueError("no legal move!")

            if win_index != 0:
                render_str(board, STATE_SIZE)
                print("win is ", win_index, "in episode", episode + 1)

                if win_index == 1:
                    reward_black = 1
                elif win_index == 2:
                    reward_black = -1
                else:
                    reward_black = 0

                for i in range(len(samples)):
                    memory.append(
                        NameTag(samples[i][0],
                                samples[i][1],
                                Tensor([reward_black]))
                    )
                agent.reset()
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

    agent = Player(STATE_SIZE, NUM_MCTS, IN_PLANES)
    agent.model = PVNet(N_BLOCKS, IN_PLANES, OUT_PLANES, STATE_SIZE)
    if use_cuda:
        agent.model.cuda()

    for i in range(1000):
        print('-----------------------------------------')
        print(i + 1, 'th training process')
        print('-----------------------------------------')
        self_play(num_episode=3)
        train(num_iter=3)
        if (i + 1) % 100 == 0:
            torch.save(
                agent.model.state_dict(),
                'models/{}train_model.pickle'.format(100 * BATCH_SIZE * 3)
            )
