import numpy as np
import torch

import agents
import model
import utils

# env_small: 9x9, env_regular: 15x15
from env import env_small as game

#WebAPI
import logging
import threading
import flask
from webapi import web_api
from webapi import game_info
from webapi import player_agent_info
from webapi import enemy_agent_info
from info.agent_info import AgentInfo
from info.game_info import GameInfo

BOARD_SIZE = game.Return_BoardParams()[0]

N_BLOCKS_PLAYER = 10
N_BLOCKS_ENEMY = 10

IN_PLANES_PLAYER = 5  # history * 2 + 1
IN_PLANES_ENEMY = 5

OUT_PLANES_PLAYER = 128
OUT_PLANES_ENEMY = 128

N_MCTS = 400
N_MATCH = 3

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

# ==================== input model path ================= #
#       'human': human play       'puct': PUCB MCTS       #
#       'uct': UCB MCTS           'random': random        #
#       'web': web play                                   #
# ======================================================= #
# example)

player_model_path = 'web'
enemy_model_path = './data/180927_9400_297233_step_model.pickle'
monitor_model_path = './data/180927_9400_297233_step_model.pickle'

class Evaluator(object):
    def __init__(self):
        self.player = None
        self.enemy = None
        self.monitor = None
        pass

    def set_agents(self, model_path_a, model_path_b, model_path_m):

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
        else:
            print('load player model:', model_path_a)
            self.player = agents.ZeroAgent(BOARD_SIZE,
                                           N_MCTS,
                                           IN_PLANES_PLAYER,
                                           noise=False)
            self.player.model = model.PVNet(N_BLOCKS_PLAYER,
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
        else:
            print('load enemy model:', model_path_b)
            self.enemy = agents.ZeroAgent(BOARD_SIZE,
                                          N_MCTS,
                                          IN_PLANES_ENEMY,
                                          noise=False)
            self.enemy.model = model.PVNet(N_BLOCKS_ENEMY,
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

        #monitor agent
        self.monitor = agents.ZeroAgent(BOARD_SIZE,
                                        N_MCTS,
                                        IN_PLANES_ENEMY,
                                        noise=False)
        self.monitor.model = model.PVNet(N_BLOCKS_ENEMY,
                                        IN_PLANES_ENEMY,
                                        OUT_PLANES_ENEMY,
                                        BOARD_SIZE).to(device)
        state_b = self.monitor.model.state_dict()
        my_state_b = torch.load(
            model_path_m, map_location='cuda:0' if use_cuda else 'cpu')
        for k, v in my_state_b.items():
            if k in state_b:
                state_b[k] = v
        self.monitor.model.load_state_dict(state_b)            

    def get_action(self, root_id, board, turn, enemy_turn):
        if turn != enemy_turn:
            if isinstance(self.player, agents.ZeroAgent):
                pi = self.player.get_pi(root_id, tau=0)
            else:
                pi = self.player.get_pi(root_id, board, turn, tau=0)
        else:
            if isinstance(self.enemy, agents.ZeroAgent):
                pi = self.enemy.get_pi(root_id, tau=0)
            else:
                pi = self.enemy.get_pi(root_id, board, turn, tau=0)

        action, action_index = utils.argmax_onehot(pi)

        return action, action_index

    def return_env(self):
        return self.env

    def reset(self):
        self.player.reset()
        self.enemy.reset()

    def put_action(self, action_idx, turn, enemy_turn):
        
        print(self.player)

        if turn != enemy_turn:
            if type(self.player) is agents.WebAgent:
                self.player.put_action(action_idx)
        else:
            if type(self.enemy) is agents.WebAgent:
                self.enemy.put_action(action_idx)


def elo(player_elo, enemy_elo, p_winscore, e_winscore):
    elo_diff = enemy_elo - player_elo
    ex_pw = 1 / (1 + 10**(elo_diff / 400))
    ex_ew = 1 / (1 + 10**(-elo_diff / 400))
    player_elo += 32 * (p_winscore - ex_pw)
    enemy_elo += 32 * (e_winscore - ex_ew)

    return player_elo, enemy_elo

evaluator = Evaluator()

def main():
    evaluator.set_agents(player_model_path, enemy_model_path, monitor_model_path)

    env = evaluator.return_env()

    result = {'Player': 0, 'Enemy': 0, 'Draw': 0}
    turn = 0
    enemy_turn = 1
    player_elo = 1500
    enemy_elo = 1500

    game_info.enemy_turn = enemy_turn
    game_info.game_status = 0

    print('Player ELO: {:.0f}, Enemy ELO: {:.0f}'.format(
        player_elo, enemy_elo))

    for i in range(N_MATCH):
        board = np.zeros([BOARD_SIZE, BOARD_SIZE])
        root_id = (0,)
        win_index = 0
        action_index = None

        game_info.game_board = board

        if i % 2 == 0:
            print('Player Color: Black')
        else:
            print('Player Color: White')

        game_info.game_status = 0 #0:Running 1:Player Win, 2: Enemy Win 3: Draw

        while win_index == 0:
            utils.render_str(board, BOARD_SIZE, action_index)
            action, action_index = evaluator.get_action(root_id,
                                                        board,
                                                        turn,
                                                        enemy_turn)

            if turn != enemy_turn:
                # player turn
                root_id = evaluator.player.root_id + (action_index,)
            else:
                # enemy turn
                root_id = evaluator.enemy.root_id + (action_index,)

            board, check_valid_pos, win_index, turn, _ = env.step(action)

            game_info.game_board = board
            game_info.action_index = int(action_index)
            game_info.win_index = win_index
            game_info.curr_turn = turn # 0 black 1 white  
            
            move = np.count_nonzero(board)
            p, v = evaluator.monitor.get_pv(root_id)

            if turn == enemy_turn:
                player_agent_info.visit = evaluator.player.get_visit()
                player_agent_info.p = evaluator.player.get_policy()   
                player_agent_info.add_value(move, v)                         
                evaluator.enemy.del_parents(root_id)

            else:
                enemy_agent_info.visit = evaluator.enemy.get_visit()
                enemy_agent_info.p = evaluator.enemy.get_policy()
                enemy_agent_info.add_value(move, v)                
                evaluator.player.del_parents(root_id)                

            if win_index != 0:
                player_agent_info.clear_values()
                enemy_agent_info.clear_values()

                game_info.game_status = win_index # 0:Running 1:Player Win, 2: Enemy Win 3: Draw

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

                game_info.enemy_turn = enemy_turn
                game_info.curr_turn = turn

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

# WebAPI
app = flask.Flask(__name__)
app.register_blueprint(web_api)
log = logging.getLogger('werkzeug')
log.disabled = True

@app.route('/action')
def action():

    action_idx = int(flask.request.args.get("action_idx"))
    data = {"success": False}
    evaluator.put_action(action_idx, game_info.curr_turn, game_info.enemy_turn)
    data["success"] = True

    return flask.jsonify(data)

if __name__ == '__main__':
    print('cuda:', use_cuda)
    np.set_printoptions(suppress=True)
    np.random.seed(0)
    torch.manual_seed(0)
    if use_cuda:
        torch.cuda.manual_seed_all(0)

    # WebAPI
    print("Activate WebAPI...")
    app_th = threading.Thread(target=app.run,
                              kwargs={"host": "0.0.0.0", "port": 5000})
    app_th.start()
    main()


