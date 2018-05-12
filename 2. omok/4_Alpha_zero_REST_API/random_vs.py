from utils import valid_actions, check_win, render_str
import numpy as np
import random
import time

import sys
sys.path.append("env/")
import env_small as game

class Random_vs:
    def main(self):
        # Game mode: 'text', 'pygame'
        game_mode = 'pygame'
        env = game.GameState(game_mode)
        state_size, action_size = game.Return_BoardParams()
        win_mark = 5

        board_shape = [state_size, state_size]
        game_board = np.zeros(board_shape)

        # 0: O, 1: X
        turn = 0
        ai_turn = 1
        turn_str = ['Black', 'White']
        alphabet_list = ['A','B','C','D','E','F','G','H','I','J',
                         'K','L','M','N','O','P','Q','R','S']

        print("\n--> Player's turn <--\n")

        while True:
            # Initialize action
            action = np.zeros([action_size])

            # AI randomly plays
            if turn == ai_turn:
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

            # Take action and get info. for update
            game_board, check_valid_pos, win_index, turn, coord = env.step(action)

            # If one move is done
            if check_valid_pos:
                last_action = state_size * coord[0] + coord[1]
                render_str(game_board, state_size, last_action)

                if turn != ai_turn:
                    print("\n--> Player's turn <--\n")

            # If game is finished
            if win_index != 0:
                game_board = np.zeros([state_size, state_size])

                # Human wins!
                if turn == ai_turn:
                    print('------------- Player Win!! -------------')
                    ai_turn = 0
                else:
                    print('------------- AI Win!! -------------')
                    ai_turn = 1
                    turn = 0

            # Delay for visualization
            time.sleep(0.01)

if __name__ == '__main__':
    agent = Random_vs()
    agent.main()
