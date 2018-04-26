'''
Author : Woonwon Lee, Jungdae Kim
Data : 2018.03.12, 2018.03.28
Project : Make your own Alpha Zero
'''
from utils import render_str, get_state_pt, get_action
from neural_net import PVNet
import numpy as np
from collections import deque
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import env_small as game
from agent import Player

STATE_SIZE = 9
N_BLOCKS = 20
IN_PLANES = 5  # history * 2 + 1
OUT_PLANES = 128
BATCH_SIZE = 32
TOTAL_ITER = 10000
N_MCTS = 800
TAU_THRES = 8
N_EPISODES = 200
N_EPOCHS = 4
SAVE_CYCLE = 1
LR = 1e-2
L2 = 1e-4


def self_play(n_episodes):
    for episode in range(n_episodes):
        print('playing {}th episode by self-play'.format(episode + 1))
        env = game.GameState('text')
        board = np.zeros([STATE_SIZE, STATE_SIZE])
        samples = []
        turn = 0
        win_index = 0
        step = 0
        action_index = None

        while win_index == 0:
            render_str(board, STATE_SIZE, action_index)
            # ====================== start mcts ============================
            pi = agent.get_pi(board, turn)
            print('\nPi:')
            print(pi.reshape(STATE_SIZE, STATE_SIZE).round(decimals=3))
            # ===================== collect samples ========================
            state = get_state_pt(agent.root_id, turn, STATE_SIZE, IN_PLANES)
            state_input = Variable(Tensor([state]))
            samples.append((state, pi))
            # ====================== print evaluation ======================
            p, v = agent.model(state_input)
            print(
                "\nProbability:\n{}".format(
                    np.exp(p.data.cpu().numpy()[0]).reshape(
                        STATE_SIZE, STATE_SIZE).round(decimals=3)))

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
            # =========================== step =============================
            board, _, win_index, turn, _ = env.step(action)
            step += 1

            # used for debugging
            # if not check_valid_pos:
            #     raise ValueError("no legal move!")

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

                render_str(board, STATE_SIZE, action_index)
                print("{} win in episode {}".format(win_color, episode + 1))
            # ====================== store in memory =======================
                for i in range(len(samples)):
                    memory.appendleft(
                        (samples[i][0], samples[i][1], reward_black))
                agent.reset()


STEPS = 0


def train(n_epochs):
    global STEPS
    global LR

    if 12e6 <= STEPS < 18e6:
        LR = 1e-3
    if STEPS >= 18e6:
        LR = 1e-4

    print('memory size:', len(memory))
    print('learning rate:', LR)

    dataloader = DataLoader(memory,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=use_cuda)
    optimizer = optim.SGD(agent.model.parameters(),
                          lr=LR,
                          momentum=0.9,
                          weight_decay=L2)

    for epoch in range(n_epochs):
        running_loss = 0.

        for i, (s, pi, z) in enumerate(dataloader):
            if use_cuda:
                s_batch = Variable(s.float()).cuda()
                pi_batch = Variable(pi.float()).cuda()
                z_batch = Variable(z.float()).cuda()
            else:
                s_batch = Variable(s.float())
                pi_batch = Variable(pi.float())
                z_batch = Variable(z.float())

            p_batch, v_batch = agent.model(s_batch)

            loss = F.mse_loss(v_batch, z_batch) + \
                torch.mean(torch.sum(-pi_batch * p_batch, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            STEPS += 1

            if (i + 1) % (N_EPISODES//2) == 0:
                print('{:3} step loss: {:.3f}'.format(
                    STEPS, running_loss / (i + 1)))


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    use_cuda = torch.cuda.is_available()
    print('cuda:', use_cuda)
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    memory = deque(maxlen=25600)
    agent = Player(STATE_SIZE, N_MCTS, IN_PLANES)
    agent.model = PVNet(N_BLOCKS, IN_PLANES, OUT_PLANES, STATE_SIZE)

    if use_cuda:
        agent.model.cuda()

    for i in range(TOTAL_ITER):
        print('-----------------------------------------')
        print('{}th training process'.format(i + 1))
        print('-----------------------------------------')

        self_play(N_EPISODES)
        train(N_EPOCHS)

        if (i + 1) % SAVE_CYCLE == 0:
            torch.save(
                agent.model.state_dict(),
                '{}_step_model.pickle'.format(STEPS))
