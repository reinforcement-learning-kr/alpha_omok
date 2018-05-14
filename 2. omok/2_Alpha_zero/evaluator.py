'''
Author : Woonwon Lee, Jungdae Kim, Kyushik Min
Data : 2018.03.12, 2018.03.28, 2018.05.14
Project : Make your own Alpha Zero
'''
from utils import render_str, get_state_pt, get_action, valid_actions
from neural_net import PVNet
import numpy as np
from collections import deque
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from agent import Player
import matplotlib.pyplot as plt

import sys
sys.path.append("env/")
import env_small as game

STATE_SIZE = 9
N_BLOCKS = 10
IN_PLANES = 5  # history * 2 + 1
OUT_PLANES = 128
BATCH_SIZE = 16
TOTAL_ITER = 10
N_MCTS = 400
TAU_THRES = 8
N_EPISODES = 1
N_EPOCHS = 1
SAVE_CYCLE = 1000
LR = 1e-3
L2 = 1e-4

def self_play(n_episodes):
    for episode in range(n_episodes):
        print('playing {}th episode by self-play'.format(episode + 1))
        env = game.GameState('text')
        board = np.zeros([STATE_SIZE, STATE_SIZE])
        samples = []
        turn = 0
        enemy_turn = 1
        win_index = 0
        step = 0
        action_index = None

        while win_index == 0:
            render_str(board, STATE_SIZE, action_index)
            if turn != enemy_turn:
                # ====================== start mcts ============================
                pi = agent.get_pi(board, turn)
                print('\nPi:')
                print(pi.reshape(STATE_SIZE, STATE_SIZE).round(decimals=2))
                # ===================== collect samples ========================
                state = get_state_pt(agent.root_id, turn, STATE_SIZE, IN_PLANES)
                state_input = Variable(Tensor([state]))
                samples.append((state, pi))
                # ====================== print evaluation ======================
                p, v = agent.model(state_input)
                print(
                    "\nProbability:\n{}".format(
                        p.data.cpu().numpy()[0].reshape(
                            STATE_SIZE, STATE_SIZE).round(decimals=2)))

                if turn == 0:
                    print("\nBlack's winrate: {:.2f}%".format(
                        (v.data[0] + 1) / 2 * 100))
                else:
                    print("\nWhite's winrate: {:.2f}%".format(
                        100 - ((v.data[0] + 1) / 2 * 100)))
                # ======================== get action ==========================
                if step < TAU_THRES:
                    tau = 1
                else:
                    tau = 0
                action, action_index = get_action(pi, tau)
                agent.root_id += (action_index,)
            else:
                # Enemy action (for now it's random)
                valid_action_list = valid_actions(board)
                valid_index = np.random.randint(len(valid_action_list))
                action = np.zeros([STATE_SIZE**2])
                action_index = valid_action_list[valid_index][1]
                action[action_index] = 1

                agent.root_id += (action_index,)

            # =========================== step =============================
            board, _, win_index, turn, _ = env.step(action)
            step += 1

            if win_index != 0:
                if win_index == 1:
                    reward_black = 1.
                    win_color = 'Black'
                elif win_index == 2:
                    reward_black = -1.
                    win_color = 'White'
                else:
                    reward_black = 0.
                    win_color = 'None'

                if turn == enemy_turn:
                    # Model wins!!
                    enemy_turn = 0
                else:
                    # Model loses!!
                    enemy_turn = 1
                    turn = 0

                render_str(board, STATE_SIZE, action_index)
                print("{} win in episode {}".format(win_color, episode + 1))

                agent.reset()

    # plt.plot(n_game, np.average(loss_list), hold=True, marker='*', ms=5)
    # plt.draw()
    # plt.pause(0.000001)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    use_cuda = torch.cuda.is_available()
    print('cuda:', use_cuda)
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    agent = Player(STATE_SIZE, N_MCTS, IN_PLANES)
    agent.model = PVNet(N_BLOCKS, IN_PLANES, OUT_PLANES, STATE_SIZE)

    if use_cuda:
        agent.model.cuda()

    torch.load('./models/2018-05-14_16_8_0_step_model.pickle')

    for i in range(TOTAL_ITER):
        print('-----------------------------------------')
        print('{}th training process'.format(i + 1))
        print('-----------------------------------------')

        self_play(N_EPISODES)
