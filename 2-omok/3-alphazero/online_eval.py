import logging

import torch

import agents
import neural_net
import utils


BOARD_SIZE = 9
N_BLOCKS = 10
IN_PLANES = 5  # history * 2 + 1
OUT_PLANES = 128
N_MCTS = 400

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


class Evaluator:
    def __init__(self, model_path_a, model_path_b):
        self.model_path_b = model_path_b

        if model_path_a:
            print('load current model:', model_path_a)
            logging.warning('load current model: {}'.format(model_path_a))
            self.player = agents.ZeroAgent(BOARD_SIZE,
                                           N_MCTS,
                                           IN_PLANES,
                                           noise=False)
            self.player.model = neural_net.NoisyPVNet(N_BLOCKS,
                                                      IN_PLANES,
                                                      OUT_PLANES,
                                                      BOARD_SIZE,
                                                      sigma_zero=0).to(device)
            # self.player.model = PVNet(N_BLOCKS,
            #                           IN_PLANES,
            #                           OUT_PLANES,
            #                           BOARD_SIZE).to(device)
            # self.player.model = PVNetW(IN_PLANES, BOARD_SIZE).to(device)

            state_a = self.player.model.state_dict()
            state_a.update(torch.load(
                model_path_a, map_location='cuda: 0' if use_cuda else 'cpu'))
            self.player.model.load_state_dict(state_a)
        else:
            self.player = agents.ZeroAgent(BOARD_SIZE, N_MCTS, IN_PLANES)
            self.player.model = neural_net.PVNet(N_BLOCKS,
                                                 IN_PLANES,
                                                 OUT_PLANES,
                                                 BOARD_SIZE).to(device)

        if model_path_b == 'random':
            print('load best model:', model_path_b)
            logging.warning('load best model: {}'.format(model_path_b))

            self.enemy = agents.RandomAgent(BOARD_SIZE)

        elif model_path_b:
            print('load best model:', model_path_b)
            self.enemy = agents.ZeroAgent(BOARD_SIZE,
                                          N_MCTS,
                                          IN_PLANES,
                                          noise=False)
            self.enemy.model = neural_net.NoisyPVNet(N_BLOCKS,
                                                     IN_PLANES,
                                                     OUT_PLANES,
                                                     BOARD_SIZE,
                                                     sigma_zero=0).to(device)
            # self.enemy.model = neural_net.PVNet(N_BLOCKS,
            #                                     IN_PLANES,
            #                                     OUT_PLANES,
            #                                     BOARD_SIZE).to(device)
            # self.enemy.model = PVNetW(IN_PLANES, BOARD_SIZE).to(device)

            state_b = self.enemy.model.state_dict()
            state_b.update(torch.load(
                model_path_b, map_location='cuda: 0' if use_cuda else 'cpu'))
            self.enemy.model.load_state_dict(state_b)

        else:
            self.enemy = agents.ZeroAgent(BOARD_SIZE, N_MCTS, IN_PLANES)
            self.enemy.model = neural_net.PVNet(N_BLOCKS,
                                                IN_PLANES,
                                                OUT_PLANES,
                                                BOARD_SIZE).to(device)

    def get_action(self, root_id, board, turn, enemy_turn):
        if turn != enemy_turn:
            self.player.model.eval()
            state = utils.get_state_pt(root_id, BOARD_SIZE, IN_PLANES)
            state_input = torch.tensor([state]).to(device).float()
            p, v = self.player.model(state_input)
            p = p.data[0].cpu().numpy()
            action, action_index = utils.get_action_eval(p, board)
        else:
            if self.model_path_b == 'random':
                pi = self.enemy.get_pi(root_id, board, turn, tau=1)
                action, action_index = utils.get_action_eval(pi, board)
            elif self.model_path_b == 'puct':
                pi = self.enemy.get_pi(root_id, board, turn, tau=0.01)
                action, action_index = utils.get_action_eval(pi, board)
            else:
                self.enemy.model.eval()
                state = utils.get_state_pt(root_id, BOARD_SIZE, IN_PLANES)
                state_input = torch.tensor([state]).to(device).float()
                p, v = self.enemy.model(state_input)
                p = p.data[0].cpu().numpy()
                action, action_index = utils.get_action_eval(p, board)

        return action, action_index

    def reset(self):
        self.player.reset()
        self.enemy.reset()
