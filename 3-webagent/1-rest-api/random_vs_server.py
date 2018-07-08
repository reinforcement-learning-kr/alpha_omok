from utils import valid_actions, check_win, render_str
import numpy as np
import random
import time

import sys
sys.path.append("env/")
import env_small as game

import flask

import threading

app = flask.Flask(__name__)

class Random_vs_server:
    
    def init(self):              

        self.env_lock = threading.Lock()

        # Game mode: 'text', 'pygame'
        game_mode = 'pygame'
        self.env = game.GameState(game_mode)
        
    def step(self, action_idx = None):
        
        state_size, action_size = game.Return_BoardParams()
        action = np.zeros([action_size])

        if action_idx != None and action_idx != '-1':
            action_idx = int(action_idx)
            action[action_idx] = 1
        
        self.env_lock.acquire()
            
        game_board, check_valid_pos, win_index, curr_turn, coord = self.env.step(action)
        
        self.env_lock.release()
            
        return game_board, check_valid_pos, win_index, curr_turn, coord
     
    def run(self):
        
        state_size, action_size = game.Return_BoardParams()
        win_mark = 5

        board_shape = [state_size, state_size]
        game_board = np.zeros(board_shape)

        # 0: O, 1: X
        curr_turn = 0
        ai_turn = 1
        turn_str = ['Black', 'White']
        alphabet_list = ['A','B','C','D','E','F','G','H','I','J',
                         'K','L','M','N','O','P','Q','R','S']  
        print("\n--> Player's turn <--\n")

        while True:
            # Initialize action
            action = np.zeros([action_size])

            # AI randomly plays
            if curr_turn == ai_turn:
                print("\n--> AI's turn <--\n")

                # Get Random Action
                valid_action_list = valid_actions(game_board)
                action_index = np.random.randint(len(valid_action_list))
                random_action = valid_action_list[action_index][1]
                action_row = int(random_action/state_size)
                action_col = int(random_action%state_size)

                print('AI Stone Index: ' + '(row: ' + str(action_row+1) +
                      ' , col: ' + alphabet_list[action_col] + ')')

                action[random_action] = 1

            # if game_mode == 'text' and turn != ai_turn:
            #     render_str(game_board, state_size, last_action)
            #     row_num = int(input('Please type row index: '))
            #     col_num = int(input('Please type col index: '))
            #     action_idx = (row_num-1) * state_size + (col_num-1)
            #     action[action_idx] = 1

            self.env_lock.acquire()

            # Take action and get info. for update
            game_board, check_valid_pos, win_index, curr_turn, coord = self.env.step(action)
    
            self.env_lock.release()
            
            # If one move is done
            if check_valid_pos:
                last_action = state_size * coord[0] + coord[1]
                render_str(game_board, state_size, last_action)

                if curr_turn != ai_turn:
                    print("\n--> Player's turn <--\n")

            # If game is finished
            if win_index != 0:
                game_board = np.zeros([state_size, state_size])

                # Human wins!
                if curr_turn == ai_turn:
                    print('------------- Player Win!! -------------')
                    ai_turn = 0
                else:
                    print('------------- AI Win!! -------------')
                    ai_turn = 1
                    curr_turn = 0

            # # Delay for visualization
            # time.sleep(0.01)

@app.route('/step')
def step():
    
    action_idx = flask.request.args.get("action_idx")
    
    data = {"success": False}

    game_board, check_valid_pos, win_index, curr_turn, coord = agent.step(action_idx)
    
    data["win_index"] = win_index
    data["curr_turn"] = curr_turn
    
    data["game_board_size"] = game_board.shape[0]
    
    game_board = game_board.reshape(game_board.size).astype(int)
    
    data["game_board_values"] = game_board.tolist()
    data["success"] = True

    return flask.jsonify(data)

if __name__ == '__main__':
    
    print("Initialize Agent...")
    agent = Random_vs_server()
    agent.init()
    
    print("Run Application...")    
    app_th = threading.Thread(target=app.run, kwargs={"host":"0.0.0.0", "port":5000})
    app_th.start()
    
    print("Run Agenet...")
    agent.run()