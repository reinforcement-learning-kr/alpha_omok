import numpy as np
import torch

import agents
from env import env_small
from neural_net import PVNet
import utils


BOARD_SIZE = 9
N_BLOCKS = 20
IN_PLANES = 3  # history * 2 + 1
OUT_PLANES = 64
N_MCTS = 400
N_MATCH = 30


class Evaluator(object):

    def __init__(self, model_path_a, model_path_b):

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
            self.player = agents.HumanAgent(BOARD_SIZE)

        elif model_path_a:
            print('load player model:', model_path_a)
            self.player = agents.ZeroAgent(BOARD_SIZE, N_MCTS, IN_PLANES)
            self.player.model = PVNet(
                N_BLOCKS, IN_PLANES, OUT_PLANES, BOARD_SIZE)

            if use_cuda:
                self.player.model.cuda()
            self.player.model.load_state_dict(torch.load(model_path_a))

        else:
            print('load player model:', model_path_a)
            self.player = agents.ZeroAgent(BOARD_SIZE, N_MCTS, IN_PLANES)
            self.player.model = PVNet(
                N_BLOCKS, IN_PLANES, OUT_PLANES, BOARD_SIZE)

            if use_cuda:
                self.player.model.cuda()

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
            self.enemy = agents.HumanAgent(BOARD_SIZE)

        elif model_path_b:
            print('load enemy model:', model_path_b)
            self.enemy = agents.ZeroAgent(BOARD_SIZE, N_MCTS, IN_PLANES)
            self.enemy.model = PVNet(
                N_BLOCKS, IN_PLANES, OUT_PLANES, BOARD_SIZE)

            if use_cuda:
                self.enemy.model.cuda()
            self.enemy.model.load_state_dict(torch.load(model_path_b))

        else:
            print('load enemy model:', model_path_b)
            self.enemy = agents.ZeroAgent(BOARD_SIZE, N_MCTS, IN_PLANES)
            self.enemy.model = PVNet(
                N_BLOCKS, IN_PLANES, OUT_PLANES, BOARD_SIZE)

            if use_cuda:
                self.enemy.model.cuda()

    def get_action(self, root_id, board, turn, enemy_turn):

        if turn != enemy_turn:
            pi = self.player.get_pi(root_id, board, turn, tau=0.01)
            action, action_index = utils.get_action(pi)
            # print(pi.reshape(BOARD_SIZE, BOARD_SIZE).round(decimals=2))
        else:
            pi = self.enemy.get_pi(root_id, board, turn, tau=0.01)
            action, action_index = utils.get_action(pi)
            # print(pi.reshape(BOARD_SIZE, BOARD_SIZE).round(decimals=2))

        return action, action_index

    def reset(self):
        self.player.reset()
        self.enemy.reset()


def main():
    print("CUDA:", use_cuda)

    # =========================== input model path ======================== #
    #    'human': human play    'random': random    None: raw model MCTS    #
    #    'puct': PUCT MCTS      'uct': UCT MCTS                             #
    # ===================================================================== #

    player_model_path = 'uct'
    enemy_model_path = 'puct'

    # ===================================================================== #

    evaluator = Evaluator(player_model_path, enemy_model_path)
    env = env_small.GameState('text')
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
            print("Player Color: Black")
        else:
            print("Player Color: White")

        while win_index == 0:
            utils.render_str(board, BOARD_SIZE, action_index)
            action, action_index = evaluator.get_action(
                root_id, board, turn, enemy_turn)

            if turn != enemy_turn:
                # print("player turn")
                root_id = evaluator.player.root_id + (action_index,)
                # evaluator.enemy.root_id = node_id

            else:
                # print("enemy turn")
                root_id = evaluator.enemy.root_id + (action_index,)
                # evaluator.player.root_id = node_id

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
    use_cuda = torch.cuda.is_available()
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    main()
