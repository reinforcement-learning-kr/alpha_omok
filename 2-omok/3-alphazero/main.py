from collections import deque
from datetime import datetime
import pickle

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import agents
from env import env_small as game
import neural_net
import utils

# Game
BOARD_SIZE = 9
N_MCTS = 400
TAU_THRES = 6

# Net
N_BLOCKS = 10
IN_PLANES = 5  # history * 2 + 1
OUT_PLANES = 64

# Training
TOTAL_ITER = 1000000
N_SELFPLAY = 400
MEMORY_SIZE = 240000
SAMPLE_SIZE = 1
N_EPOCHS = 1200
BATCH_SIZE = 32
LR = 1e-4
L2 = 1e-4

# Data
DATASET_SAVE = True


def self_play(n_selfplay):
    state_black = deque()
    state_white = deque()
    pi_black = deque()
    pi_white = deque()

    resign_val_balck = []
    resign_val_white = []
    resign_val = []
    resign_v = -1.0
    n_resign_thres = N_SELFPLAY // 10

    for episode in range(n_selfplay):
        env = game.GameState('text')
        board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        turn = 0
        root_id = (0,)
        win_index = 0
        resign_index = 0
        time_step = 0
        action_index = None

        while win_index == 0:
            utils.render_str(board, BOARD_SIZE, action_index)
            # ====================== start mcts ============================ #

            if time_step < TAU_THRES:
                tau = 1
            else:
                tau = 1e-2

            pi = Agent.get_pi(root_id, board, turn, tau)

            print('\nPi:')
            print(pi.reshape(BOARD_SIZE, BOARD_SIZE).round(decimals=2))

            # ===================== collect samples ======================== #

            state = utils.get_state_pt(root_id, BOARD_SIZE, IN_PLANES)

            if turn == 0:
                state_black.appendleft(state)
                pi_black.appendleft(pi)
            else:
                state_white.appendleft(state)
                pi_white.appendleft(pi)

            # ====================== print evaluation ====================== #

            state_input = Variable(Tensor([state]))
            p, v = Agent.model(state_input)

            print(
                "\nProb:\n{}".format(
                    p.data.cpu().numpy()[0].reshape(
                        BOARD_SIZE, BOARD_SIZE).round(decimals=2)))

            if turn == 0:
                print("\nBlack's win%: {:.2f}%".format(
                    (v.item() + 1) / 2 * 100))

                if episode < n_resign_thres:
                    resign_val_balck.append(v.item())

                elif v.item() < resign_v:
                    resign_index = 2
                    print('"Black Resign!"')
            else:
                print("\nWhite's win%: {:.2f}%".format(
                    (v.item() + 1) / 2 * 100))

                if episode < n_resign_thres:
                    resign_val_white.append(v.item())

                elif v.item() < resign_v:
                    resign_index = 1
                    print('"White Resign!"')

            # ======================== get action ========================== #

            action, action_index = utils.get_action(pi)
            root_id += (action_index,)

            # =========================== step ============================= #

            board, _, win_index, turn, _ = env.step(action)
            time_step += 1

            if resign_index != 0:
                win_index = resign_index
                result['Resign'] += 1

            if win_index != 0:

                if win_index == 1:
                    reward_black = 1.
                    reward_white = -1.
                    result['Black'] += 1

                    if episode < n_resign_thres:
                        resign_val.append(min(resign_val_balck))
                        resign_val_balck.clear()
                        resign_val_white.clear()

                elif win_index == 2:
                    reward_black = -1.
                    reward_white = 1.
                    result['White'] += 1

                    if episode < n_resign_thres:
                        resign_val.append(min(resign_val_white))
                        resign_val_white.clear()
                        resign_val_balck.clear()

                else:
                    reward_black = 0.
                    reward_white = 0.
                    result['Draw'] += 1

                    if episode < n_resign_thres:
                        resign_val.append(min(resign_val_balck))
                        resign_val.append(min(resign_val_white))
                        resign_val_balck.clear()
                        resign_val_white.clear()

                if episode + 1 == n_resign_thres:
                    resign_v = min(resign_val)
                    resign_val.clear()

                print('Resign win%: {:.2f}%'.format((resign_v + 1) / 2 * 100))

            # ====================== store in memory ======================= #
                while state_black or state_white:
                    if state_black:
                        memory.appendleft((state_black.pop(),
                                           pi_black.pop(),
                                           reward_black))
                    if state_white:
                        memory.appendleft((state_white.pop(),
                                           pi_white.pop(),
                                           reward_white))

            # =========================  result  =========================== #

                utils.render_str(board, BOARD_SIZE, action_index)
                bw, ww, dr, rs = result['Black'], result['White'], \
                    result['Draw'], result['Resign']
                print('')
                print('=' * 20,
                      " {:3} Game End   ".format(episode + 1),
                      '=' * 20)
                print('Black Win: {:3}   '
                      'White Win: {:3}   '
                      'Draw: {:2}   '
                      'Win%: {:.2f}%'
                      '\nResign: {:2}'.format(
                          bw, ww, dr,
                          (bw + 0.5 * dr) / (bw + ww + dr) * 100,
                          rs))
                print('memory size:', len(memory))

                Agent.reset()


def train(n_epochs, n_iter):
    global step
    global writer

    Agent.model.train()

    loss_all = []
    loss_v = []
    loss_p = []

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

        for i, (s, pi, z) in enumerate(dataloader):
            if i == SAMPLE_SIZE:
                break

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
            p_loss = -(pi_batch * p_batch.log()).sum(dim=1).mean()

            loss = v_loss + p_loss

            loss_v.append(v_loss.item())
            loss_p.append(p_loss.item())
            loss_all.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # tensorboad & print loss
            writer.add_scalar('Loss', loss.item(), step)
            writer.add_scalar('Loss V', v_loss.item(), step)
            writer.add_scalar('Loss P', p_loss.item(), step)

            print('{:4} step Loss: {:.4f}   '
                  'Loss V: {:.4f}   '
                  'Loss P: {:.4f}'.format(
                      step, loss.item(), v_loss.item(), p_loss.item()))

            step += 1

        writer.add_scalar('mean Loss', np.mean(loss_all), n_iter)
        writer.add_scalar('mean Loss V', np.mean(loss_v), n_iter)
        writer.add_scalar('mean Loss P', np.mean(loss_p), n_iter)

        print('-' * 58)
        print('mean Loss: {:.4f} '
              'mean Loss V: {:.4f} '
              'mean Loss P: {:.4f}'.format(np.mean(loss_all),
                                           np.mean(loss_v),
                                           np.mean(loss_p)))
        print('-' * 58)


def save_data(memory, step, n_iter):
    datetime_now = datetime.now().strftime('%y%m%d_%H%M')

    # save model
    torch.save(
        Agent.model.state_dict(),
        'data/{}_{}_{}_step_model.pickle'.format(datetime_now, n_iter, step))

    if DATASET_SAVE:
        with open('data/{}_{}_{}_step_dataset.pickle'.format(
                datetime_now, n_iter, step), 'wb') as f:
            pickle.dump(memory, f, pickle.HIGHEST_PROTOCOL)


def load_data(model_path, dataset_path):
    global memory, step, start_iter

    if model_path:
        print('load model: {}\n'.format(model_path))
        Agent.model.load_state_dict(torch.load(model_path))
        step = int(model_path.split('_')[3])
        start_iter = int(model_path.split('_')[2]) + 1

    if dataset_path:
        print('load dataset: {}\n'.format(dataset_path))
        with open(dataset_path, 'rb') as f:
            memory = pickle.load(f)
            memory = deque(memory, maxlen=MEMORY_SIZE)


def reset_iter(memory, result, n_iter):
    result['Black'] = 0
    result['White'] = 0
    result['Draw'] = 0
    result['Resign'] = 0
    # memory.clear()


if __name__ == '__main__':
    # numpy printing style
    np.set_printoptions(suppress=True)

    # set random seeds
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # global variable
    memory = deque(maxlen=MEMORY_SIZE)
    writer = SummaryWriter()
    result = {'Black': 0, 'White': 0, 'Draw': 0, 'Resign': 0}
    step = 0
    start_iter = 1

    # gpu or cpu
    use_cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    print('CUDA:', use_cuda)

    # init agent & model
    # Agent = agents.PUCTAgent(BOARD_SIZE, N_MCTS)
    Agent = agents.ZeroAgent(BOARD_SIZE, N_MCTS, IN_PLANES)
    Agent.model = neural_net.PVNet(N_BLOCKS,
                                   IN_PLANES,
                                   OUT_PLANES,
                                   BOARD_SIZE)

    if use_cuda:
        Agent.model.cuda()

# ====================== self-play & training ====================== #

    model_path = None
    dataset_path = None

    load_data(model_path, dataset_path)

    for n_iter in range(start_iter, TOTAL_ITER + 1):
        print('=' * 20, " {:4} Iteration ".format(n_iter), '=' * 20)

        if dataset_path:
            train(N_EPOCHS, n_iter)
            save_data(memory, step, n_iter)
            self_play(N_SELFPLAY)
            reset_iter(memory, result, n_iter)
        else:
            self_play(N_SELFPLAY)
            train(N_EPOCHS, n_iter)
            save_data(memory, step, n_iter)
            reset_iter(memory, result, n_iter)
