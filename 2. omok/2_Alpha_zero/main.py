from collections import deque
import datetime
import pickle
# from pprint import pprint

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import agents
from env import env_small as game
from neural_net import PVNet
import utils


BOARD_SIZE = 9
N_BLOCKS = 20
IN_PLANES = 3  # history * 2 + 1
OUT_PLANES = 64
BATCH_SIZE = 32
TOTAL_ITER = 1000000
N_MCTS = 400
TAU_THRES = 8
N_EPISODES = 1
N_EPOCHS = 1
LR = 2e-3
L2 = 1e-4
MEMORY_SIZE = 8000
TRAIN_START_SIZE = 8000
SAVE_CYCLE = 250
ONLINE_TRAIN = False  # True: Update every episode when training begins
EMPTY_MEMORY = True   # whether to empty memory after saving the memory


def self_play():
    env = game.GameState('text')
    board = np.zeros((BOARD_SIZE, BOARD_SIZE))
    samples_black = []
    samples_white = []
    turn = 0
    root_id = (0,)
    win_index = 0
    step = 0
    action_index = None

    while win_index == 0:
        utils.render_str(board, BOARD_SIZE, action_index)
        # ====================== start mcts ============================ #

        pi = Agent.get_pi(root_id, board, turn)
        print('\nPi:')
        print(pi.reshape(BOARD_SIZE, BOARD_SIZE).round(decimals=2))

        # ===================== collect samples ======================== #

        state = utils.get_state_pt(root_id, BOARD_SIZE, IN_PLANES)
        state_input = Variable(Tensor([state]))

        if turn == 0:
            samples_black.append((state, pi))
        else:
            samples_white.append((state, pi))

        # ====================== print evaluation ====================== #

        p, v = Agent.model(state_input)
        print(
            "\nProbability:\n{}".format(
                p.data.cpu().numpy()[0].reshape(
                    BOARD_SIZE, BOARD_SIZE).round(decimals=2)))

        if turn == 0:
            print("\nBlack's win%: {:.2f}%".format(
                (v.data[0] + 1) / 2 * 100))
        else:
            print("\nWhite's win%: {:.2f}%".format(
                (v.data[0] + 1) / 2 * 100))

        # ======================== get action ========================== #

        if step < TAU_THRES:
            tau = 1
        else:
            tau = 0
        action, action_index = utils.get_action(pi, tau)
        root_id += (action_index,)

        # =========================== step ============================= #

        board, _, win_index, turn, _ = env.step(action)
        step += 1

        if win_index != 0:

            if win_index == 1:
                reward_black = 1.
                reward_white = -1.
                result['Black'] += 1

            elif win_index == 2:
                reward_black = -1.
                reward_white = 1.
                result['White'] += 1

            else:
                reward_black = 0.
                reward_white = 0.
                result['Draw'] += 1

        # ====================== store in memory ======================= #

            for i in range(len(samples_black)):
                memory.append((samples_black[i][0],
                               samples_black[i][1],
                               reward_black))

            for j in range(len(samples_white)):
                memory.append((samples_white[j][0],
                               samples_white[j][1],
                               reward_white))

        # =========================  result  =========================== #

            utils.render_str(board, BOARD_SIZE, action_index)
            Agent.reset()
            bw, ww, dr = result['Black'], result['White'], result['Draw']
            print('')
            print('=' * 18,
                  " {:4} Iteration End ".format(bw + ww + dr),
                  '=' * 18)
            print('Black Win: {:3}  '
                  'White Win: {:3}  '
                  'Draw: {}  '
                  'Win%: {:.2f}%'.format(
                      bw, ww, dr,
                      (bw + 0.5 * dr) / (bw + ww + dr) * 100))
            print('memory size:', len(memory))
            # pprint(memory)


def train(n_epochs):
    global step
    global loss_pv, loss_p, loss_v
    # global LR

    # if 3000 < step <= 6000:
    #     LR = 2e-4
    # if 6000 < step <= 9000:
    #     LR = 2e-5
    # if step > 9000:
    #     LR = 2e-6

    print('=' * 20, ' Start Learning ', '=' * 20)
    print('learning rate:', LR)

    optimizer = optim.SGD(Agent.model.parameters(),
                          lr=LR,
                          momentum=0.9,
                          weight_decay=L2)

    for epoch in range(n_epochs):
        dataloader = DataLoader(memory,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=use_cuda)

        for (s, pi, z) in dataloader:
            if use_cuda:
                s_batch = Variable(s.float()).cuda()
                pi_batch = Variable(pi.float()).cuda()
                z_batch = Variable(z.float()).cuda()
            else:
                s_batch = Variable(s.float())
                pi_batch = Variable(pi.float())
                z_batch = Variable(z.float())

            p_batch, v_batch = Agent.model(s_batch)

            v_loss = F.mse_loss(v_batch, z_batch)
            p_loss = torch.mean(torch.sum(-pi_batch * p_batch.log(), 1))
            loss = v_loss + p_loss

            # steps.append(step)
            loss_v.append(v_loss.item())
            loss_p.append(p_loss.item())
            loss_pv.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            print('{} step Loss: {:.4f}   '
                  'VLoss: {:.4f}   '
                  'PLoss: {:.4f}'.format(
                      step, loss.item(), v_loss.item(), p_loss.item()))

            if ONLINE_TRAIN:
                break

    print('-' * 58)
    print('min Loss: {:.4f}  '
          'min VLoss: {:.4f}  '
          'min PLoss: {:.4f}'.format(min(loss_pv), min(loss_v), min(loss_p)))

    # plt.plot(STEPS, V_LOSS, marker='o', ms=3, label='V Loss')
    # plt.plot(STEPS, P_LOSS, marker='o', ms=3, label='P Loss')
    # plt.plot(STEPS, LOSS, marker='*', ms=5, label='Total Loss')
    # plt.legend()
    # plt.ylabel('Loss')
    # plt.xlabel('Step')
    # plt.grid(True, ls='--', lw=.5, c='k', alpha=.3)
    # plt.show()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    use_cuda = torch.cuda.is_available()
    print('cuda:', use_cuda)
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    memory = deque(maxlen=MEMORY_SIZE)
    Agent = agents.ZeroAgent(BOARD_SIZE, N_MCTS, IN_PLANES)
    Agent.model = PVNet(N_BLOCKS, IN_PLANES, OUT_PLANES, BOARD_SIZE)
    step = 0
    # steps = []
    loss_pv = deque(maxlen=1000)
    loss_v = deque(maxlen=1000)
    loss_p = deque(maxlen=1000)
    result = {'Black': 0, 'White': 0, 'Draw': 0}

    # ==================== load model & dataset ======================== #

    model_path = None
    dataset_path = None

    if model_path:
        print('load model: {}\n'.format(model_path[5:]))
        Agent.model.load_state_dict(torch.load(model_path))
        step = int(model_path.split('_')[2])

    if dataset_path:
        print('load dataset: {}\n'.format(dataset_path))
        with open(dataset_path, 'rb') as f:
            memory = pickle.load(f)

    # =================================================================== #

    if use_cuda:
        Agent.model.cuda()

    for i in range(TOTAL_ITER):
        print('=' * 20, " {:4} Iteration ".format(i + 1), '=' * 20)
        self_play()

        if len(memory) == TRAIN_START_SIZE:
            train(N_EPOCHS)

            if step % SAVE_CYCLE == 0:
                datetime_now = datetime.datetime.now().strftime('%y%m%d_%H%M')
                # save model
                torch.save(
                    Agent.model.state_dict(),
                    'data/{}_{}_step_model.pickle'.format(datetime_now, step))
                # save dataset
                with open('data/{}_{}_step_dataset.pickle'.format(
                        datetime_now, step), 'wb') as f:
                    pickle.dump(memory, f, pickle.HIGHEST_PROTOCOL)
                # reset result
                result = {'Black': 0, 'White': 0, 'Draw': 0}

                if EMPTY_MEMORY:
                    memory.clear()
