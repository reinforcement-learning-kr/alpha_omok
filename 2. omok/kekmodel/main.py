'''
Author : Woonwon Lee, Jungdae Kim
Data : 2018.03.12, 2018.03.28
Project : Make your own Alpha Zero
'''
from utils import render_str, get_state_pt, get_action
from neural_net import PVNet
import random
import numpy as np
from collections import deque, namedtuple
import torch
import torch.optim as optim
# import torch.nn.functional as F
from torch.autograd import Variable
import env_small as game
from agent import Player

N_BLOCKS = 20
IN_PLANES = 5
OUT_PLANES = 128
BATCH_SIZE = 32
SAVE_CYCLE = 1
LR = 0.2
L2 = 0.0001

STATE_SIZE = 9
NUM_MCTS = 200


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
        action_index = None

        while win_index == 0:
            render_str(board, STATE_SIZE, action_index)
            # ======================  start mcts ========================
            pi = agent.get_pi(board, turn)
            # print('')
            # print(pi.reshape(STATE_SIZE, STATE_SIZE).round(decimals=3))
            state = get_state_pt(agent.root_id, turn, STATE_SIZE, IN_PLANES)
            state = Tensor([state])
            state_input = Variable(state)
            samples.append((state, Tensor([pi])))
            p, v = agent.model(state_input)
            if turn == 0:
                print("\nBlack's winrate: {:.1f}%".format(
                    (v.data[0, 0] + 1) / 2 * 100))
            else:
                print("\nWhite's winrate: {:.1f}%".format(
                    100 - ((v.data[0, 0] + 1) / 2 * 100)))
            print(
                "\nProbability:\n{}".format(p.data.cpu().numpy()[0].reshape(
                    STATE_SIZE, STATE_SIZE).round(decimals=3)))

            if step < tau_thres:
                tau = 1
            else:
                tau = 0
            # ======================== get_action ========================
            action, action_index = get_action(pi, tau)
            agent.root_id += (action_index,)

            # =========================== step ===========================
            board, _, win_index, turn, _ = env.step(action)
            step += 1

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

                render_str(board, STATE_SIZE, action_index)
                print("win is ", win_color, "in episode", episode + 1)

                for i in range(len(samples)):
                    memory.appendleft(
                        NameTag(samples[i][0],
                                samples[i][1],
                                Tensor([reward_black])))
                agent.reset()


def train(num_iter):
    optimizer = optim.SGD(
        agent.model.parameters(), lr=LR, momentum=0.9, weight_decay=L2)
    running_loss = 0.
    print('memory size:', len(memory))
    for i in range(num_iter):
        batch = random.sample(memory, BATCH_SIZE)
        batch = NameTag(*zip(*batch))

        s_batch = Variable(torch.cat(batch.s))
        pi_batch = Variable(torch.cat(batch.pi))
        z_batch = Variable(torch.cat(batch.z))
        p_batch, v_batch = agent.model(s_batch)

        loss = torch.sum((z_batch - v_batch)**2 -
                         torch.bmm(
            pi_batch.view(BATCH_SIZE, 1, STATE_SIZE**2),
            torch.log(p_batch.view(BATCH_SIZE, STATE_SIZE**2, 1))).view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        print('{:3} iterarion loss: {:.3f}'.format(
            i + 1, running_loss / (i + 1)))


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
    num_episode = 30
    num_iter = 25
    for i in range(100):
        print('-----------------------------------------')
        print(i + 1, 'th training process')
        print('-----------------------------------------')
        self_play(num_episode)
        train(num_iter)
        if (i + 1) % SAVE_CYCLE == 0:
            torch.save(
                agent.model.state_dict(),
                '{}train_model.pickle'.format(
                    SAVE_CYCLE * BATCH_SIZE * num_iter))
