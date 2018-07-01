from collections import deque
from datetime import datetime
import logging
import pickle
import random

import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import agents
from env import env_small as game
import neural_net
import online_eval
import utils

logging.basicConfig(
    filename='logs/log_{}.txt'.format(datetime.now().strftime('%y%m%d')),
    level=logging.WARNING)
logging.warning(datetime.now().isoformat())

# Game
BOARD_SIZE = 9
N_MCTS = 400
TAU_THRES = 6
RESIGN_MODE = True
PRINT_SELFPLAY = True
agents.PRINT_MCTS = True

# Net
N_BLOCKS = 10
IN_PLANES = 7  # history * 2 + 1
OUT_PLANES = 128

# Training
TOTAL_ITER = 1000000
N_SELFPLAY = 400
MEMORY_SIZE = 160000
N_EPOCHS = 1
N_MATCH = 400
EVAL_THRES = 15
BATCH_SIZE = 32
LR = 1e-4
L2 = 1e-4
USE_TENSORBOARD = True

# Parameter sharing
online_eval.N_BLOCKS = N_BLOCKS
online_eval.IN_PLANES = IN_PLANES
online_eval.OUT_PLANES = OUT_PLANES


def self_play(n_selfplay):
    state_black = deque()
    state_white = deque()
    pi_black = deque()
    pi_white = deque()

    if RESIGN_MODE:
        resign_val_balck = []
        resign_val_white = []
        resign_val = []
        resign_v = -1.0
        n_resign_thres = N_SELFPLAY // 10

    for episode in range(n_selfplay):
        if (episode + 1) % 50 == 0:
            logging.warning('Playing Episode {:3}'.format(episode + 1))
        env = game.GameState('text')
        board = np.zeros((BOARD_SIZE, BOARD_SIZE))
        turn = 0
        root_id = (0,)
        win_index = 0
        time_step = 0
        action_index = None

        if RESIGN_MODE:
            resign_index = 0

        while win_index == 0:
            if PRINT_SELFPLAY:
                utils.render_str(board, BOARD_SIZE, action_index)

            # ====================== start mcts ============================ #

            if time_step < TAU_THRES:
                tau = 1
            else:
                tau = 1e-2
            pi = Agent.get_pi(root_id, board, turn, tau)

            if PRINT_SELFPLAY:
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

            Agent.model.eval()
            state_input = torch.FloatTensor([state], device=device)
            p, v = Agent.model(state_input)

            if PRINT_SELFPLAY:
                print(
                    '\nProb:\n{}'.format(
                        p.data.cpu().numpy()[0].reshape(
                            BOARD_SIZE, BOARD_SIZE).round(decimals=2)))

            if turn == 0:
                if PRINT_SELFPLAY:
                    print("\nBlack's win%: {:.2f}%".format(
                        (v.item() + 1) / 2 * 100))
                if RESIGN_MODE:
                    if episode < n_resign_thres:
                        resign_val_balck.append(v.item())
                    elif v.item() < resign_v:
                        resign_index = 2
                        if PRINT_SELFPLAY:
                            print('"Black Resign!"')
            else:
                if PRINT_SELFPLAY:
                    print("\nWhite's win%: {:.2f}%".format(
                        (v.item() + 1) / 2 * 100))
                if RESIGN_MODE:
                    if episode < n_resign_thres:
                        resign_val_white.append(v.item())
                    elif v.item() < resign_v:
                        resign_index = 1
                        if PRINT_SELFPLAY:
                            print('"White Resign!"')

            # ======================== get action ========================== #

            action, action_index = utils.get_action(pi)
            root_id += (action_index,)

            # =========================== step ============================= #

            board, _, win_index, turn, _ = env.step(action)
            time_step += 1
            if RESIGN_MODE:
                if resign_index != 0:
                    win_index = resign_index
                    result['Resign'] += 1

            if win_index != 0:
                if win_index == 1:
                    reward_black = 1.
                    reward_white = -1.
                    result['Black'] += 1

                    if RESIGN_MODE:
                        if episode < n_resign_thres:
                            for val in resign_val_balck:
                                resign_val.append(val)
                            resign_val_balck.clear()
                            resign_val_white.clear()

                elif win_index == 2:
                    reward_black = -1.
                    reward_white = 1.
                    result['White'] += 1

                    if RESIGN_MODE:
                        if episode < n_resign_thres:
                            for val in resign_val_white:
                                resign_val.append(val)
                            resign_val_white.clear()
                            resign_val_balck.clear()
                else:
                    reward_black = 0.
                    reward_white = 0.
                    result['Draw'] += 1

                    if RESIGN_MODE:
                        if episode < n_resign_thres:
                            for val in resign_val_balck:
                                resign_val.append(val)
                            for val in resign_val_white:
                                resign_val.append(val)
                            resign_val_balck.clear()
                            resign_val_white.clear()

                if RESIGN_MODE:
                    if episode + 1 == n_resign_thres:
                        # resign_val.sort()
                        # idx = (len(resign_val) // 200) - 1
                        # resign_v = resign_val[idx]
                        resign_v = min(resign_val)
                        resign_val.clear()

                    if PRINT_SELFPLAY:
                        print('Resign win%: {:.2f}%'.format(
                            (resign_v + 1) / 2 * 100))

            # ====================== store in memory ======================= #
                while state_black or state_white:
                    if state_black:
                        cur_memory.appendleft((state_black.pop(),
                                               pi_black.pop(),
                                               reward_black))
                    if state_white:
                        cur_memory.appendleft((state_white.pop(),
                                               pi_white.pop(),
                                               reward_white))

            # =========================  result  =========================== #
                if PRINT_SELFPLAY:
                    utils.render_str(board, BOARD_SIZE, action_index)

                bw, ww, dr, rs = result['Black'], result['White'], \
                    result['Draw'], result['Resign']

                if PRINT_SELFPLAY:
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
                    print('current memory size:', len(cur_memory))

                Agent.reset()


def train(lr, n_epochs, n_iter):
    global step
    global Writer

    Agent.model.train()
    loss_all = []
    loss_v = []
    loss_p = []

    num_sample = len(cur_memory) // 4
    train_memory = []

    if rep_memory:
        train_memory.extend(random.sample(rep_memory, num_sample))

    train_memory.extend(cur_memory)

    optimizer = optim.Adam(Agent.model.parameters(),
                           lr=lr,
                           weight_decay=L2)

    # optimizer = optim.SGD(Agent.model.parameters(),
    #                       lr=lr,
    #                       momentum=0.9,
    #                       weight_decay=L2)

    dataloader = DataLoader(train_memory,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=use_cuda)

    print('-' * 19, ' Start Learning ', '-' * 19)
    print('current memory size:', len(cur_memory))
    print('replay memory size:', len(rep_memory))
    print('train memory size:', len(train_memory))
    print('optimizer: {}'.format(optimizer))
    logging.warning('-' * 19 + ' Start Learning ' + '-' * 19)
    logging.warning('current memory size: {}'.format(len(cur_memory)))
    logging.warning('replay memory size: {}'.format(len(rep_memory)))
    logging.warning('train memory size: {}'.format(len(train_memory)))
    logging.warning('optimizer: {}'.format(optimizer))

    for epoch in range(n_epochs):
        for i, (s, pi, z) in enumerate(dataloader):
            s_batch = s.to(device).float()
            pi_batch = pi.to(device).float()
            z_batch = z.to(device).float()

            p_batch, v_batch = Agent.model(s_batch)

            v_loss = F.mse_loss(v_batch, z_batch)
            p_loss = -(pi_batch * (p_batch + 1e-5).log()).sum(dim=1).mean()
            loss = v_loss + p_loss

            loss_v.append(v_loss.item())
            loss_p.append(p_loss.item())
            loss_all.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            # tensorboad & print loss
            if USE_TENSORBOARD:
                Writer.add_scalar('Loss', loss.item(), step)
                Writer.add_scalar('Loss V', v_loss.item(), step)
                Writer.add_scalar('Loss P', p_loss.item(), step)

            if PRINT_SELFPLAY:
                print('{:4} step Loss: {:.4f}   '
                      'Loss V: {:.4f}   '
                      'Loss P: {:.4f}'.format(
                          step, loss.item(), v_loss.item(), p_loss.item()))


def eval_model(player_path, enemy_path):
    print('-' * 18, ' Start Evaluation ', '-' * 18)
    logging.warning('-' * 18 + ' Start Evaluation ' + '-' * 18)

    Agent.model.eval()
    evaluator = online_eval.Evaluator(player_path, enemy_path)

    env_eval = game.GameState('text')

    result = {'Player': 0, 'Enemy': 0, 'Draw': 0}
    turn = 0
    enemy_turn = 1

    for i in range(N_MATCH):
        board = np.zeros([BOARD_SIZE, BOARD_SIZE])
        root_id = (0,)
        evaluator.player.root_id = root_id
        evaluator.enemy.root_id = root_id
        win_index = 0
        action_index = None
        while win_index == 0:
            action, action_index = evaluator.get_action(
                root_id, board, turn, enemy_turn)
            if turn != enemy_turn:
                root_id = evaluator.player.root_id + (action_index,)
                evaluator.enemy.root_id = root_id
            else:
                root_id = evaluator.enemy.root_id + (action_index,)
                evaluator.player.root_id = root_id

            board, chk_legal_move, win_index, turn, _ = env_eval.step(action)

            # used for warningging
            if not chk_legal_move:
                print('action:', action)
                print('board:', board)
                raise ValueError('illegal move!')

            # episode end
            if win_index != 0:
                if turn == enemy_turn:
                    if win_index == 3:
                        result['Draw'] += 1
                    else:
                        result['Player'] += 1
                else:
                    if win_index == 3:
                        result['Draw'] += 1
                    else:
                        result['Enemy'] += 1

                # Change turn
                enemy_turn = abs(enemy_turn - 1)
                turn = 0

                pw, ew, dr = result['Player'], result['Enemy'], result['Draw']
                winrate = (pw + 0.5 * dr) / (pw + ew + dr) * 100

                evaluator.reset()

    winrate = (pw + 0.5 * dr) / (pw + ew + dr) * 100

    print('win%: {:.2f}%'.format(winrate))
    logging.warning('win%: {:.2f}%'.format(winrate))
    return winrate


def train_and_eval(lr, best_model_path):
    for i in range(EVAL_THRES):
        train(lr, N_EPOCHS, n_iter)
        save_model(Agent, n_iter, step)

        player_path = 'data/{}_{}_{}_step_model.pickle'.format(
            datetime_now, n_iter, step)

        winrate = eval_model(player_path, best_model_path)

        if winrate > 55:
            best_model_path = player_path
            print('Find Best Model')
            logging.warning('Find Best Model')
            success = True
            return best_model_path, success

    print('Do Not Find Best Model')
    logging.warning('Do Not Find Best Model')
    success = False
    return best_model_path, success


def train_and_eval_SGD(lr, best_model_path):
    winrates = []
    ng_count = 0
    for i in range(EVAL_THRES):
        train(lr, N_EPOCHS, n_iter)
        save_model(Agent, n_iter, step)

        player_path = 'data/{}_{}_{}_step_model.pickle'.format(
            datetime_now, n_iter, step)

        winrate = eval_model(player_path, best_model_path)
        winrates.append(winrate)

        if winrate > 55:
            best_model_path = player_path
            print('Find Best Model')
            logging.warning('Find Best Model')
            success = True
            return best_model_path, success

        if winrate < max(winrates):
            ng_count += 1

        if ng_count == 2:
            print('Reduce LR: {} -> {}'.format(lr, lr / 2))
            logging.warning('Reduce LR: {} -> {}'.format(lr, lr / 2))
            lr /= 2
            ng_count = 0

    print('Do Not Find Best Model')
    logging.warning('Do Not Find Best Model')
    success = False
    return best_model_path, success


def save_model(agent, n_iter, step):
    torch.save(
        agent.model.state_dict(),
        'data/{}_{}_{}_step_model.pickle'.format(datetime_now, n_iter, step))


def save_dataset(memory, n_iter, step):
    with open('data/{}_{}_{}_step_dataset.pickle'.format(
            datetime_now, n_iter, step), 'wb') as f:
        pickle.dump(memory, f, pickle.HIGHEST_PROTOCOL)


def load_data(model_path, dataset_path):
    global cur_memory, step, start_iter
    if model_path:
        print('load model: {}'.format(model_path))
        logging.warning('load model: {}'.format(model_path))
        Agent.model.load_state_dict(torch.load(model_path))
        step = int(model_path.split('_')[2])
        start_iter = int(model_path.split('_')[1]) + 1
    if dataset_path:
        print('load dataset: {}'.format(dataset_path))
        logging.warning('load dataset: {}'.format(dataset_path))
        with open(dataset_path, 'rb') as f:
            cur_memory = deque(pickle.load(f), maxlen=MEMORY_SIZE)


def reset_iter(memory, result, n_iter):
    result['Black'] = 0
    result['White'] = 0
    result['Draw'] = 0
    result['Resign'] = 0
    memory.clear()


if __name__ == '__main__':
    # numpy printing style
    np.set_printoptions(suppress=True)

    # set random seeds
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # global variable
    rep_memory = deque(maxlen=MEMORY_SIZE)
    cur_memory = deque()
    if USE_TENSORBOARD:
        Writer = SummaryWriter()
    result = {'Black': 0, 'White': 0, 'Draw': 0, 'Resign': 0}
    step = 0
    start_iter = 1
    total_epoch = 0

    # gpu or cpu
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('cuda:', use_cuda)

    # init agent & model
    Agent = agents.ZeroAgent(BOARD_SIZE, N_MCTS, IN_PLANES)
    Agent.model = neural_net.PVNet(N_BLOCKS,
                                   IN_PLANES,
                                   OUT_PLANES,
                                   BOARD_SIZE).to(device)

# ====================== self-play & training ====================== #
    model_path = None
    dataset_path = None
    best_model_path = None

    first_train = False

    if first_train:
        datetime_now = datetime.now().strftime('%y%m%d')
        save_model(Agent, 0, 0)
        best_model_path = 'data/{}_{}_{}_step_model.pickle'.format(
            datetime_now, 0, 0)

    load_data(model_path, dataset_path)

    for n_iter in range(start_iter, TOTAL_ITER + 1):
        print('=' * 58)
        print(' ' * 22, '{:2} Iteration'.format(n_iter), ' ' * 22)
        print('=' * 58)
        logging.warning('=' * 58)
        logging.warning(
            ' ' * 20 + "  {:2} Iteration  ".format(n_iter) + ' ' * 20)
        logging.warning('=' * 58)

        datetime_now = datetime.now().strftime('%y%m%d')

        if dataset_path:
            best_model_path, success = train_and_eval(LR, best_model_path)
            if success:
                rep_memory.extend(cur_memory)
            else:
                print('Load the Previous Best Model')
                logging.warning('Load the Previous Best Model')
                load_data(best_model_path, dataset_path=False)
            save_dataset(rep_memory, n_iter, step)
            reset_iter(cur_memory, result, n_iter)
            self_play(N_SELFPLAY)
        else:
            self_play(N_SELFPLAY)
            best_model_path, success = train_and_eval(LR, best_model_path)
            if success:
                rep_memory.extend(cur_memory)
            else:
                print('Load the Previous Best Model')
                logging.warning('Load the Previous Best Model')
                load_data(best_model_path, dataset_path=False)
            save_dataset(rep_memory, n_iter, step)
            reset_iter(cur_memory, result, n_iter)
