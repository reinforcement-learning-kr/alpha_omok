from utils import render_str, get_action, get_state_pt
from agent import Player, PUCTAgent, RandomAgent
from neural_net import PVNet
from torch.autograd import Variable
import numpy as np
import torch
import sys
sys.path.append("env/")
USE_CUDA = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
STATE_SIZE = 9
N_BLOCKS = 3
IN_PLANES = 5
OUT_PLANES = 32
N_MCTS = 400
N_MATCH = 30


class Evaluator:
    def __init__(self, model_path_a, model_path_b):
        self.model_path_b = model_path_b
        if model_path_a == 'puct':
            print('load player model:', model_path_a)
            self.player = UCTAgent(STATE_SIZE, N_MCTS)
        elif model_path_a:
            print('load player model:', model_path_a)
            self.player = Player(STATE_SIZE, N_MCTS, IN_PLANES)
            self.player.model = PVNet(IN_PLANES, STATE_SIZE)
            self.player.model.load_state_dict(torch.load(model_path_a))
        else:
            self.player = Player(STATE_SIZE, N_MCTS, IN_PLANES)
            self.player.model = PVNet(IN_PLANES, STATE_SIZE)

        if model_path_b == 'human':
            print('load enemy model:', model_path_b)
            self.enemy = HumanAgent(STATE_SIZE)

        elif model_path_b == 'puct':
            print('load enemy model:', model_path_b)
            self.enemy = PUCTAgent(STATE_SIZE, N_MCTS)

        elif model_path_b == 'random':
            print('load enemy model:', model_path_b)
            self.enemy = RandomAgent(STATE_SIZE)

        elif model_path_b:
            print('load enemy model:', model_path_b)
            self.enemy = Player(STATE_SIZE, N_MCTS, IN_PLANES)
            self.enemy.model = PVNet(IN_PLANES, STATE_SIZE)
            self.enemy.model.load_state_dict(torch.load(model_path_b))

        else:
            self.enemy = Player(STATE_SIZE, N_MCTS, IN_PLANES)
            self.enemy.model = PVNet(IN_PLANES, STATE_SIZE)

        if USE_CUDA:
            if model_path_a != 'puct':
                self.player.model.cuda()
            if model_path_b != 'random' and model_path_b != 'puct':
                self.enemy.model.cuda()

    def get_action(self, root_id, board, turn, enemy_turn):
        if turn != enemy_turn:
            self.player.model.eval()
            state = get_state_pt(root_id, STATE_SIZE, IN_PLANES)
            state_input = Variable(Tensor([state]))
            p, v = self.player.model(state_input)
            p = p.data[0].cpu().numpy()
            action, action_index = get_action(p, board)
        else:
            if self.model_path_b == 'random':
                pi = self.enemy.get_pi(root_id, board, turn)
                action, action_index = get_action(pi, board)
            elif self.model_path_b == 'puct':
                pi = self.enemy.get_pi(root_id, board, turn)
                action, action_index = get_action(pi, board)
            else:
                self.enemy.model.eval()
                state = get_state_pt(root_id, STATE_SIZE, IN_PLANES)
                state_input = Variable(Tensor([state]))
                p, v = self.enemy.model(state_input)
                p = p.data[0].cpu().numpy()
                action, action_index = get_action(p, board)

        return action, action_index

    def reset(self):
        self.player.reset()
        self.enemy.reset()


def main():
    import env_small as game
    print("CUDA:", USE_CUDA)

    # ========================== input model path ======================= #
    # 'random': no MCTS, 'puct': model free MCTS, None: random model MCTS
    # player_model_path = 'models/model_236.pickle'
    player_model_path = 'models/model_45_0607.pickle'
    enemy_model_path = 'puct'

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
            render_str(board, STATE_SIZE, action_index)
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

                render_str(board, STATE_SIZE, action_index)

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
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    main()
