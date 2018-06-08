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
TAU_THRES = 8

# Net
N_BLOCKS = 10
IN_PLANES = 5  # history * 2 + 1
OUT_PLANES = 64

# Training
TOTAL_ITER = 1000000
N_SELFPLAY = 400
N_EPOCHS = 1
BATCH_SIZE = 32
LR = 0.0015
L2 = 1e-4

# Data
DATASET_SAVE = True


def self_play(n_selfplay):
    state_black = deque()
    state_white = deque()
    pi_black = deque()
    pi_white = deque()

    for episode in range(n_selfplay):
        env = game.GameState('text')
        board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        turn = 0
        root_id = (0,)
        win_index = 0
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
            state_input = Variable(Tensor([state]))

            if turn == 0:
                s_sym, pi_sym = utils.symmetry_choice(state, pi)
                state_black.appendleft(s_sym.copy())
                pi_black.appendleft(pi_sym.copy())
            else:
                s_sym, pi_sym = utils.symmetry_choice(state, pi)
                state_white.appendleft(s_sym.copy())
                pi_white.appendleft(pi_sym.copy())

            # ====================== print evaluation ====================== #

            p, v = Agent.model(state_input)

            print(
                "\nLogit:\n{}".format(
                    p.data.cpu().numpy()[0].reshape(
                        BOARD_SIZE, BOARD_SIZE).round(decimals=2)))

            if turn == 0:
                print("\nBlack's win%: {:.2f}%".format(
                    (v.data[0] + 1) / 2 * 100))
            else:
                print("\nWhite's win%: {:.2f}%".format(
                    (v.data[0] + 1) / 2 * 100))

            # ======================== get action ========================== #

            action, action_index = utils.get_action(pi)
            root_id += (action_index,)

            # =========================== step ============================= #

            board, _, win_index, turn, _ = env.step(action)
            time_step += 1

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
                while state_black and state_white:
                    memory.appendleft((state_black.pop(),
                                       pi_black.pop(),
                                       reward_black))
                    memory.appendleft((state_white.pop(),
                                       pi_white.pop(),
                                       reward_white))

            # =========================  result  =========================== #

                utils.render_str(board, BOARD_SIZE, action_index)
                bw, ww, dr = result['Black'], result['White'], result['Draw']
                print('')
                print('=' * 20,
                      " {:3} Game End   ".format(episode + 1),
                      '=' * 20)
                print('Black Win: {:3}   '
                      'White Win: {:3}   '
                      'Draw: {:3}   '
                      'Win%: {:.2f}%'.format(
                          bw, ww, dr,
                          (bw + 0.5 * dr) / (bw + ww + dr) * 100))
                print('memory size:', len(memory))

                Agent.reset()


def train(n_epochs, n_iter):
    global step
    global writer

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


def save_data(memory, n_iter, step):
    datetime_now = datetime.now().strftime('%y%m%d_%H%M')

    # save model
    torch.save(
        Agent.model.state_dict(),
        'data/{}_{}_{}_step_model.pickle'.format(datetime_now, n_iter, step))

    if DATASET_SAVE:
        with open('data/{}_{}_step_dataset.pickle'.format(
                datetime_now, step), 'wb') as f:
            pickle.dump(memory, f, pickle.HIGHEST_PROTOCOL)


def load_data(model_path, dataset_path):
    global memory, step, start_iter

    if model_path:
        print('load model: {}\n'.format(model_path))
        Agent.model.load_state_dict(torch.load(model_path))
        step = int(model_path.split('_')[2])
        start_iter = int(model_path.split('_')[1]) + 1

    if dataset_path:
        print('load dataset: {}\n'.format(dataset_path))
        with open(dataset_path, 'rb') as f:
            memory = pickle.load(f)
            memory = deque(memory, maxlen=200000)


def reset_iter(memory, result, n_iter):
    result['Black'] = 0
    result['White'] = 0
    result['Draw'] = 0
    # memory.clear()


if __name__ == '__main__':
    # numpy printing style
    np.set_printoptions(suppress=True)

    # set random seeds
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # global variable
    memory = deque(maxlen=200000)
    writer = SummaryWriter()
    result = {'Black': 0, 'White': 0, 'Draw': 0}
    start_iter = 0
    step = 0

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

    for n_iter in range(start_iter, TOTAL_ITER):
        print('=' * 20, " {:4} Iteration ".format(n_iter), '=' * 20)

        if dataset_path:
            train(N_EPOCHS, n_iter)
            save_data(memory, n_iter, step)
            self_play(N_SELFPLAY)
            reset_iter(memory, result, n_iter)
        else:
            self_play(N_SELFPLAY)
            train(N_EPOCHS, n_iter)
            save_data(memory, n_iter, step)
            reset_iter(memory, result, n_iter)
