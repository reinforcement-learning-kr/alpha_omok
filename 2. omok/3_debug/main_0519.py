'''
Author : Woonwon Lee, Jungdae Kim, Kyushik Min
Data : 2018.03.12, 2018.03.28, 2018.05.11
Project : Make your own Alpha Zero
Objective : find the problem of code. Let's Debugging!!
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
# import matplotlib.pyplot as plt
import datetime

import sys
sys.path.append("env/")
import env_small as game
from evaluator import Evaluator

STATE_SIZE = 9
N_BLOCKS = 3
IN_PLANES = 5  # history * 2 + 1
OUT_PLANES = 32
BATCH_SIZE = 16
TOTAL_ITER = 100000
N_MCTS = 400
TAU_THRES = 8
N_EPISODES = 10
N_EPOCHS = 10
SAVE_CYCLE = 1000
LR = 1e-3
L2 = 1e-4
N_MATCH = 10


def self_play(n_episodes):
    for episode in range(n_episodes):
        # print('playing {}th episode by self-play'.format(episode + 1))
        env = game.GameState('text')
        board = np.zeros([STATE_SIZE, STATE_SIZE])
        samples = []
        turn = 0
        root_id = (0,)
        win_index = 0
        step = 0
        action_index = None

        while win_index == 0:
            # render_str(board, STATE_SIZE, action_index)

            # ====================== start mcts ============================
            # pi = agent.get_pi(board, turn)
            # print('\nPi:')
            # print(pi.reshape(STATE_SIZE, STATE_SIZE).round(decimals=2))
            # ===================== collect samples ========================
            state = get_state_pt(root_id, turn, STATE_SIZE, IN_PLANES)
            state_input = Variable(Tensor([state]))

            # ====================== print evaluation ======================
            p, v = agent.model(state_input)

            '''
            print(
                "\nProbability:\n{}".format(
                    p.data.cpu().numpy()[0].reshape(
                        STATE_SIZE, STATE_SIZE).round(decimals=2)))

            if turn == 0:
                print("\nBlack's winrate: {:.2f}%".format(
                    (v.data[0].numpy()[0] + 1) / 2 * 100))
            else:
                print("\nWhite's winrate: {:.2f}%".format(
                    100 - ((v.data[0].numpy()[0] + 1) / 2 * 100)))
            '''
            # ======================== get action ==========================
            p = p.data[0].cpu().numpy()
            action, action_index = get_action(p, board)

            root_id += (action_index,)
            # =========================== step =============================
            board, check_valid_pos, win_index, turn, _ = env.step(action)
            if turn == 1:
                samples.append((state, action))
            step += 1

            # used for debugging
            if not check_valid_pos:
                raise ValueError("no legal move!")

            if win_index != 0:
                if win_index == 1:
                    reward_black = 1.
                    # win_color = 'Black'
                elif win_index == 2:
                    reward_black = -1.
                    # win_color = 'White'
                else:
                    reward_black = 0.
                    # win_color = 'None'

                # render_str(board, STATE_SIZE, action_index)
                # print("{} win in episode {}".format(win_color, episode + 1))
            # ====================== store in memory =======================
                for i in range(len(samples)):
                    memory.append((samples[i][0], samples[i][1], reward_black))
                agent.reset()


STEPS = 0


def train(n_game, n_epochs):
    global STEPS
    # global LR

    # if 12e6 <= STEPS < 18e6:
    #     LR = 1e-3
    # if STEPS >= 18e6:
    #     LR = 1e-4

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

    loss_list = []

    for epoch in range(n_epochs):
        running_loss = 0.

        for i, (s, a, z) in enumerate(dataloader):
            if use_cuda:
                s_batch = Variable(s.float()).cuda()
                a_batch = Variable(a.float()).cuda()
                z_batch = Variable(z.float()).cuda()
            else:
                s_batch = Variable(s.float())
                a_batch = Variable(a.float())
                z_batch = Variable(z.float())

            p_batch, v_batch = agent.model(s_batch)
            v_batch = v_batch.view(v_batch.size(0))
            loss_v = F.mse_loss(v_batch, z_batch)

            p_action = a_batch * p_batch
            p_action = torch.sum(p_action, 1)
            loss_p = torch.mean(
                torch.sum(-z_batch * torch.log(p_action + 1e-5)))
            loss = loss_v + loss_p
            loss_list.append(loss.data[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            STEPS += 1

            if (i + 1) % (N_EPISODES) == 0:
                print('{:3} step loss: {:.3f}'.format(
                    STEPS, running_loss / (i + 1)))
    torch.save(
        agent.model.state_dict(),
        './models/model_{}.pickle'.format(n_game))

    # plt.plot(n_game, np.average(loss_list), hold=True, marker='*', ms=5)
    # plt.draw()
    # plt.pause(0.000001)


def eval_model(player_model_path, enemy_model_path):
    evaluator = Evaluator(player_model_path, enemy_model_path)

    env = game.GameState('text')
    result = {'Player': 0, 'Enemy': 0, 'Draw': 0}
    turn = 0
    enemy_turn = 1
    player_elo = 1500
    enemy_elo = 1500
    print('Player ELO: {:.0f}, Enemy ELO: {:.0f}'.format(
        player_elo, enemy_elo))

    for i in range(N_MATCH):
        board = np.zeros([STATE_SIZE, STATE_SIZE])
        root_id = (0,)
        evaluator.player.root_id = root_id
        evaluator.enemy.root_id = root_id
        win_index = 0
        action_index = None
        if i % 2 == 0:
            print("Player Color: Black")
        else:
            print("Player Color: White")

        while win_index == 0:
            # render_str(board, STATE_SIZE, action_index)
            action, action_index = evaluator.get_action(
                root_id, board, turn, enemy_turn)

            if turn != enemy_turn:
                # print("player turn")
                root_id = evaluator.player.root_id + (action_index,)
                evaluator.enemy.root_id = root_id
            else:
                # print("enemy turn")
                root_id = evaluator.enemy.root_id + (action_index,)
                evaluator.player.root_id = root_id

            board, check_valid_pos, win_index, turn, _ = env.step(action)

            # used for debugging
            if not check_valid_pos:
                raise ValueError("no legal move!")

            if win_index != 0:

                if turn == enemy_turn:
                    if win_index == 3:
                        result['Draw'] += 1
                        print("\nDraw!")

                        elo_diff = enemy_elo - player_elo
                        ex_pw = 1 / (1 + 10**(elo_diff / 400))
                        ex_ew = 1 / (1 + 10**(-elo_diff / 400))
                        player_elo += 32 * (0.5 - ex_pw)
                        enemy_elo += 32 * (0.5 - ex_ew)

                    else:
                        result['Player'] += 1
                        print("\nPlayer Win!")

                        elo_diff = enemy_elo - player_elo
                        ex_pw = 1 / (1 + 10**(elo_diff / 400))
                        ex_ew = 1 / (1 + 10**(-elo_diff / 400))
                        player_elo += 32 * (1 - ex_pw)
                        enemy_elo += 32 * (0 - ex_ew)
                else:
                    if win_index == 3:
                        result['Draw'] += 1
                        print("\nDraw!")

                        elo_diff = enemy_elo - player_elo
                        ex_pw = 1 / (1 + 10**(elo_diff / 400))
                        ex_ew = 1 / (1 + 10**(-elo_diff / 400))
                        player_elo += 32 * (0.5 - ex_pw)
                        enemy_elo += 32 * (0.5 - ex_ew)
                    else:
                        result['Enemy'] += 1
                        print("\nEnemy Win!")

                        elo_diff = enemy_elo - player_elo
                        ex_pw = 1 / (1 + 10**(elo_diff / 400))
                        ex_ew = 1 / (1 + 10**(-elo_diff / 400))
                        player_elo += 32 * (0 - ex_pw)
                        enemy_elo += 32 * (1 - ex_ew)

                # Change turn
                enemy_turn = abs(enemy_turn - 1)
                turn = 0

                # render_str(board, STATE_SIZE, action_index)

                pw, ew, dr = result['Player'], result['Enemy'], result['Draw']
                winrate = (pw + 0.5 * dr) / (pw + ew + dr) * 100
                print('')
                print('=' * 20, " {}  Game End  ".format(i + 1), '=' * 20)
                print('Player Win: {}  Enemy Win: {}  Draw: {}  Winrate: {:.2f}%'.format(
                    pw, ew, dr, winrate))
                print('Player ELO: {:.0f}, Enemy ELO: {:.0f}'.format(
                    player_elo, enemy_elo))
                evaluator.reset()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    use_cuda = torch.cuda.is_available()
    print('cuda:', use_cuda)
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    memory = deque(maxlen=50000)
    agent = Player(STATE_SIZE, N_MCTS, IN_PLANES)
    agent.model = PVNet(IN_PLANES, STATE_SIZE)

    datetime_now = str(datetime.date.today()) + '_' + \
        str(datetime.datetime.now().hour) + '_' + \
        str(datetime.datetime.now().minute)

    if use_cuda:
        agent.model.cuda()

    for i in range(TOTAL_ITER):
        print('-----------------------------------------')
        print('{}th training process'.format(i + 1))
        print('-----------------------------------------')

        self_play(N_EPISODES)
        train(i, N_EPOCHS)

        player_model_path = "./models/model_{}.pickle".format(i)
        if i == 0:
            enemy_model_path = 'random'
        else:
            enemy_model_path = "./models/model_{}.pickle".format(i-1)

        eval_model(player_model_path, enemy_model_path)

        '''
        if (i + 1) >= 160:
            train(i, N_EPOCHS)

        if (i + 1) % SAVE_CYCLE == 0:
            torch.save(
                agent.model.state_dict(),
                './models/{}_{}_step_model.pickle'.format(datetime_now, STEPS))

        '''
