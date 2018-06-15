from utils import render_str, get_action
from agent import Player, PUCTAgent, RandomAgent
from neural_net import PVNet
import numpy as np
import torch
import sys
sys.path.append("env/")
USE_CUDA = torch.cuda.is_available()

STATE_SIZE = 9
N_BLOCKS = 10
IN_PLANES = 9
OUT_PLANES = 64
N_MCTS = 100
N_MATCH = 30


class Evaluator:
    def __init__(self, model_path_a, model_path_b):
        self.player = Player(STATE_SIZE, N_MCTS, IN_PLANES)

        if model_path_a:
            print('load player model:', model_path_a)
            self.player.model.load_state_dict(torch.load(model_path_a))
        else:
            self.player.model = PVNet(
                N_BLOCKS, IN_PLANES, OUT_PLANES, STATE_SIZE)

        if model_path_b == 'random':
            print('load enemy model:', model_path_b)
            self.enemy = RandomAgent(STATE_SIZE)

        elif model_path_b == 'puct':
            print('load enemy model:', model_path_b)
            self.enemy = PUCTAgent(STATE_SIZE, N_MCTS)

        elif model_path_b:
            print('load enemy model:', model_path_b)
            self.enemy = Player(STATE_SIZE, N_MCTS, IN_PLANES)
            self.enemy.model.load_state_dict(torch.load(model_path_b))

        else:
            self.enemy = Player(STATE_SIZE, N_MCTS, IN_PLANES)
            self.enemy.model = PVNet(
                N_BLOCKS, IN_PLANES, OUT_PLANES, STATE_SIZE)

        if USE_CUDA:
            self.player.model.cuda()
            if model_path_b != 'random' and model_path_b != 'puct':
                self.enemy.model.cuda()

    def get_action(self, i, board, turn):
        if turn == 0:
            pi = self.player.get_pi(board, turn)
            action, action_index = get_action(pi, tau=0)
        else:
            pi = self.enemy.get_pi(board, turn)
            action, action_index = get_action(pi, tau=0)

        return action, action_index

    def reset(self):
        self.player.reset()
        self.enemy.reset()


def main():
    import env_small as game
    print("CUDA:", USE_CUDA)

    # input model path
    # 'random': no MCTS, 'puct': model free MCTS, None: random model MCTS
    player_model_path = None
    enemy_model_path = None

    evaluator = Evaluator(player_model_path, enemy_model_path)

    env = game.GameState('text')
    result = {'Player': 0, 'Enemy': 0, 'Draw': 0}

    enemy_turn = 1

    for i in range(N_MATCH):
        board = np.zeros([STATE_SIZE, STATE_SIZE])
        turn = 0
        win_index = 0
        action_index = None

        while win_index == 0:
            render_str(board, STATE_SIZE, action_index)
            action, action_index = evaluator.get_action(i, board, turn)

            if turn != enemy_turn:
                print("player turn")
                node_id = evaluator.player.root_id + (action_index,)
                evaluator.enemy.root_id = node_id
            else:
                print("enemy turn")
                node_id = evaluator.enemy.root_id + (action_index,)
                evaluator.player.root_id = node_id

            board, check_valid_pos, win_index, turn, _ = env.step(action)

            # used for debugging
            if not check_valid_pos:
                raise ValueError("no legal move!")

            if win_index != 0:
                if turn == enemy_turn:
                    if win_index == 3:
                        result['Draw'] += 1
                        print("Draw!")
                    else:
                        result['Player'] += 1
                        print("Player Win!")
                else:
                    if win_index == 3:
                        result['Draw'] += 1
                        print("Draw!")
                    else:
                        result['Enemy'] += 1
                        print("Enemy Win!")

                # Change turn
                enemy_turn = abs(enemy_turn - 1)
                turn = 0

                render_str(board, STATE_SIZE, action_index)
                pw, ew, dr = result['Player'], result['Enemy'], result['Draw']
                winrate = (pw + 0.5 * dr) / (pw + ew + dr) * 100
                print('')
                print('=' * 20, " {}  Game End  ".format(i + 1), '=' * 20)
                print('Player Win: {}  Enemy Win: {}  Draw: {}  \
                       Winrate: {:.2f}%'.format(pw, ew, dr, winrate))
                evaluator.reset()


if __name__ == '__main__':
    main()
