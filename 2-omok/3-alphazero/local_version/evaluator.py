"""
env_small: 9x9
env_regular: 15x15
"""
import numpy as np
import torch

import agents
from env import env_small as game
import neural_net
import utils

BOARD_SIZE = game.Return_BoardParams()[0]

N_BLOCKS_PLAYER = 10
N_BLOCKS_ENEMY = 10

IN_PLANES_PLAYER = 5  # history * 2 + 1
IN_PLANES_ENEMY = 5

OUT_PLANES_PLAYER = 128
OUT_PLANES_ENEMY = 128

N_MCTS = 3000
N_MATCH = 12

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# =========================== input model path ======================== #
#   'human': human play   'random': random     None: raw model MCTS     #
#   'puct': PUCT MCTS     'uct': UCT MCTS     'web': human web player   #
# ===================================================================== #

player_model_path = None
enemy_model_path = None

# ===================================================================== #


class Evaluator(object):
    def __init__(self, model_path_a, model_path_b):

        if model_path_a == 'human' or model_path_b == 'human':
            game_mode = 'pygame'
        else:
            game_mode = 'text'

        self.env = game.GameState(game_mode)

        if model_path_a == 'random':
            print('load player model:', model_path_a)
            self.player = agents.RandomAgent(BOARD_SIZE)
        elif model_path_a == 'puct':
            print('load player model:', model_path_a)
            self.player = agents.PUCTAgent(BOARD_SIZE, N_MCTS)
        elif model_path_a == 'uct':
            print('load player model:', model_path_a)
            self.player = agents.UCTAgent(BOARD_SIZE, N_MCTS)
        elif model_path_a == 'human':
            print('load player model:', model_path_a)
            self.player = agents.HumanAgent(BOARD_SIZE, self.env)
        elif model_path_a == 'web':
            print('load player model:', model_path_a)
            self.player = agents.WebAgent(BOARD_SIZE)
        elif model_path_a:
            print('load player model:', model_path_a)
            self.player = agents.ZeroAgent(BOARD_SIZE,
                                           N_MCTS,
                                           IN_PLANES_PLAYER,
                                           noise=False)
            self.player.model = neural_net.PVNet(N_BLOCKS_PLAYER,
                                                 IN_PLANES_PLAYER,
                                                 OUT_PLANES_PLAYER,
                                                 BOARD_SIZE).to(device)
            state_a = self.player.model.state_dict()
            my_state_a = torch.load(
                model_path_a, map_location='cuda:0' if use_cuda else 'cpu')
            for k, v in my_state_a.items():
                if k in state_a:
                    state_a[k] = v
            self.player.model.load_state_dict(state_a)
        else:
            print('load player model:', model_path_a)
            self.player = agents.ZeroAgent(BOARD_SIZE,
                                           N_MCTS,
                                           IN_PLANES_PLAYER,
                                           noise=False)
            self.player.model = neural_net.PVNet(N_BLOCKS_PLAYER,
                                                 IN_PLANES_PLAYER,
                                                 OUT_PLANES_PLAYER,
                                                 BOARD_SIZE).to(device)
        if model_path_b == 'random':
            print('load enemy model:', model_path_b)
            self.enemy = agents.RandomAgent(BOARD_SIZE)
        elif model_path_b == 'puct':
            print('load enemy model:', model_path_b)
            self.enemy = agents.PUCTAgent(BOARD_SIZE, N_MCTS)
        elif model_path_b == 'uct':
            print('load enemy model:', model_path_b)
            self.enemy = agents.UCTAgent(BOARD_SIZE, N_MCTS)
        elif model_path_b == 'human':
            print('load enemy model:', model_path_b)
            self.enemy = agents.HumanAgent(BOARD_SIZE, self.env)
        elif model_path_b == 'web':
            print('load enemy model:', model_path_b)
            self.enemy = agents.WebAgent(BOARD_SIZE)
        elif model_path_b:
            print('load enemy model:', model_path_b)
            self.enemy = agents.ZeroAgent(BOARD_SIZE,
                                          N_MCTS,
                                          IN_PLANES_ENEMY,
                                          noise=False)
            self.enemy.model = neural_net.PVNet(N_BLOCKS_ENEMY,
                                                IN_PLANES_ENEMY,
                                                OUT_PLANES_ENEMY,
                                                BOARD_SIZE).to(device)
            state_b = self.enemy.model.state_dict()
            my_state_b = torch.load(
                model_path_b, map_location='cuda:0' if use_cuda else 'cpu')
            for k, v in my_state_b.items():
                if k in state_b:
                    state_b[k] = v
            self.enemy.model.load_state_dict(state_b)
        else:
            print('load enemy model:', model_path_b)
            self.enemy = agents.ZeroAgent(BOARD_SIZE,
                                          N_MCTS,
                                          IN_PLANES_ENEMY,
                                          noise=False)
            self.enemy.model = neural_net.PVNet(N_BLOCKS_ENEMY,
                                                IN_PLANES_ENEMY,
                                                OUT_PLANES_ENEMY,
                                                BOARD_SIZE).to(device)

    def get_action(self, root_id, board, turn, enemy_turn):
        if turn != enemy_turn:
            pi = self.player.get_pi(root_id, board, turn, tau=0.01)
        else:
            pi = self.enemy.get_pi(root_id, board, turn, tau=0.01)

        action, action_index = utils.argmax_pi(pi)

        return action, action_index

    def return_env(self):
        return self.env

    def reset(self):
        self.player.reset()
        self.enemy.reset()


class OnlineEvaluator(Evaluator):
    def __init__(self, model_path_a, model_path_b):
        super().__init__(model_path_a, model_path_b)

    def get_action(self, root_id, board, turn, enemy_turn):
        if turn != enemy_turn:
            self.player.model.eval()
            with torch.no_grad():
                state = utils.get_state_pt(
                    root_id, BOARD_SIZE, IN_PLANES_PLAYER)
                state_input = torch.tensor([state]).to(device).float()
                p, v = self.player.model(state_input)
                p = p.cpu().numpy()[0]
            action, action_index = utils.get_action_eval(p, board)
        else:
            self.enemy.model.eval()
            with torch.no_grad():
                state = utils.get_state_pt(
                    root_id, BOARD_SIZE, IN_PLANES_ENEMY)
                state_input = torch.tensor([state]).to(device).float()
                p, v = self.enemy.model(state_input)
                p = p.cpu().numpy()[0]
            action, action_index = utils.get_action_eval(p, board)

        return action, action_index


def elo(player_elo, enemy_elo, p_winscore, e_winscore):
    elo_diff = enemy_elo - player_elo
    ex_pw = 1 / (1 + 10**(elo_diff / 400))
    ex_ew = 1 / (1 + 10**(-elo_diff / 400))
    player_elo += 32 * (p_winscore - ex_pw)
    enemy_elo += 32 * (e_winscore - ex_ew)

    return player_elo, enemy_elo


def main():
    evaluator = Evaluator(player_model_path, enemy_model_path)

    env = evaluator.return_env()

    result = {'Player': 0, 'Enemy': 0, 'Draw': 0}
    turn = 0
    enemy_turn = 1
    player_elo = 1500
    enemy_elo = 1500

    print('Player ELO: {:.0f}, Enemy ELO: {:.0f}'.format(
        player_elo, enemy_elo))

    for i in range(N_MATCH):
        board = np.zeros([BOARD_SIZE, BOARD_SIZE])
        root_id = (0,)
        win_index = 0
        action_index = None

        if i % 2 == 0:
            print('Player Color: Black')
        else:
            print('Player Color: White')

        while win_index == 0:
            utils.render_str(board, BOARD_SIZE, action_index)
            action, action_index = evaluator.get_action(
                root_id, board, turn, enemy_turn)

            if turn != enemy_turn:
                # player turn
                root_id = evaluator.player.root_id + (action_index,)
            else:
                # enemy turn
                root_id = evaluator.enemy.root_id + (action_index,)

            board, check_valid_pos, win_index, turn, _ = env.step(action)

            if turn == enemy_turn:
                evaluator.enemy.del_parents(root_id)
            else:
                evaluator.player.del_parents(root_id)

            # used for debugging
            if not check_valid_pos:
                raise ValueError('no legal move!')

            if win_index != 0:
                if turn == enemy_turn:
                    if win_index == 3:
                        result['Draw'] += 1
                        print('\nDraw!')
                        player_elo, enemy_elo = elo(
                            player_elo, enemy_elo, 0.5, 0.5)
                    else:
                        result['Player'] += 1
                        print('\nPlayer Win!')
                        player_elo, enemy_elo = elo(
                            player_elo, enemy_elo, 1, 0)
                else:
                    if win_index == 3:
                        result['Draw'] += 1
                        print('\nDraw!')
                        player_elo, enemy_elo = elo(
                            player_elo, enemy_elo, 0.5, 0.5)
                    else:
                        result['Enemy'] += 1
                        print('\nEnemy Win!')
                        player_elo, enemy_elo = elo(
                            player_elo, enemy_elo, 0, 1)

                utils.render_str(board, BOARD_SIZE, action_index)
                # Change turn
                enemy_turn = abs(enemy_turn - 1)
                turn = 0
                pw, ew, dr = result['Player'], result['Enemy'], result['Draw']
                winrate = (pw + 0.5 * dr) / (pw + ew + dr) * 100
                print('')
                print('=' * 20, " {}  Game End  ".format(i + 1), '=' * 20)
                print('Player Win: {}'
                      '  Enemy Win: {}'
                      '  Draw: {}'
                      '  Winrate: {:.2f}%'.format(
                          pw, ew, dr, winrate))
                print('Player ELO: {:.0f}, Enemy ELO: {:.0f}'.format(
                    player_elo, enemy_elo))
                evaluator.reset()


if __name__ == '__main__':
    print('cuda:', use_cuda)
    np.set_printoptions(suppress=True)
    np.random.seed(0)
    torch.manual_seed(0)

    if use_cuda:
        torch.cuda.manual_seed_all(0)

    main()
