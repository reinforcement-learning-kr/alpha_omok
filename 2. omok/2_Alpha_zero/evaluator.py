from utils import render_str, get_action, valid_actions
from agent import Player
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
N_MCTS = 400
N_MATCH = 30


class RandomAgent:
    def __init__(self, state_size):
        self.state_size = state_size

    def get_pi(self, board, turn):
        action = valid_actions(board)
        prob = 1 / len(action)
        pi = np.zeros(self.state_size**2, 'float')

        for loc, idx in action:
            pi[idx] = prob

        return pi

    def reset(self):
        pass


class PUCTAgent:
    def __init__(self, state_size, num_mcts, inplanes):
        self.state_size = state_size
        self.num_mcts = num_mcts
        self.inplanes = inplanes
        self.win_mark = 5
        self.alpha = 0.15
        self.turn = 0
        self.board = np.zeros([self.state_size, self.state_size])
        self.root_id = (0,)
        self.model = None
        self.tree = {self.root_id: {'board': self.board,
                                    'player': self.turn,
                                    'child': [],
                                    'parent': None,
                                    'n': 0.,
                                    'w': 0.,
                                    'q': 0.,
                                    'p': None}}

    def init_mcts(self, board, turn):
        self.turn = turn
        self.board = board

    def selection(self, tree):
        node_id = self.root_id

        while True:
            if node_id in tree:
                num_child = len(tree[node_id]['child'])
                # check if current node is leaf node
                if num_child == 0:
                    return node_id
                else:
                    leaf_id = node_id
                    qu = {}
                    ids = []

                    if leaf_id == self.root_id:
                        noise = np.random.dirichlet(
                            self.alpha * np.ones(num_child))

                    for i in range(num_child):
                        action = tree[leaf_id]['child'][i]
                        child_id = leaf_id + (action,)
                        n = tree[child_id]['n']
                        q = tree[child_id]['q']

                        if leaf_id == self.root_id:
                            p = tree[child_id]['p']
                            p = 0.75 * p + 0.25 * noise[i]
                        else:
                            p = tree[child_id]['p']

                        total_n = tree[tree[child_id]['parent']]['n'] - 1

                        u = 5. * p * np.sqrt(total_n) / (n + 1)

                        if tree[leaf_id]['player'] == 0:
                            qu[child_id] = q + u

                        else:
                            qu[child_id] = q - u

                    if tree[leaf_id]['player'] == 0:
                        max_value = max(qu.values())
                        ids = [key for key, value in qu.items() if value ==
                               max_value]
                        node_id = ids[np.random.choice(len(ids))]
                    else:
                        min_value = min(qu.values())
                        ids = [key for key, value in qu.items() if value ==
                               min_value]
                        node_id = ids[np.random.choice(len(ids))]
            else:
                tree[node_id] = {'board': self.board,
                                 'player': self.turn,
                                 'child': [],
                                 'parent': None,
                                 'n': 0.,
                                 'w': 0.,
                                 'q': 0.,
                                 'p': None}
                return node_id


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
            if model_path_b != 'random':
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
    enemy_model_path = 'random'

    evaluator = Evaluator(player_model_path, enemy_model_path)

    env = game.GameState('text')
    result = {'Player': 0, 'Enemy': 0, 'Draw': 0}

    for i in range(N_MATCH):
        board = np.zeros([STATE_SIZE, STATE_SIZE])
        turn = 0
        player_color = 'Black' if turn == 0 else 'White'
        win_index = 0
        action_index = None

        while win_index == 0:
            render_str(board, STATE_SIZE, action_index)
            action, action_index = evaluator.get_action(i, board, turn)

            if turn == 0:
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
                if win_index == 1:
                    if player_color == 'Black':
                        result['Player'] += 1
                        print("Player Win!")
                    else:
                        result['Enemy'] += 1
                        print("Enemy Win")
                elif win_index == 2:
                    if player_color == 'White':
                        result['Player'] += 1
                        print("Player Win!")
                    else:
                        result['Enemy'] += 1
                        print("Enemy Win")
                else:
                    result['Draw'] += 1
                    print("Draw!")

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
