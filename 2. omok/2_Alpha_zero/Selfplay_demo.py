# Deep Q-Network Algorithm

# Import modules
import pygame
import random
import numpy as np
import time

import env_small
import env_regular
import env_large as game


class self_demo:
    def __init__(self):

        # Game Information
        self.game_name = game.ReturnName()

        # Get parameters
        self.Num_action = game.Return_Num_Action()

        # Initialize Parameters
        self.step = 1
        self.score = 0
        self.episode = 0

        # Turn = 0: Black, 1: White
        self.turn = 0

        self.Num_spot = int(np.sqrt(self.Num_action))
        self.gameboard = np.zeros([self.Num_spot, self.Num_spot])
        self.state = np.zeros([self.Num_spot, self.Num_spot, 17])
        self.check_valid_pos = 0
        self.win_index = 0

    def main(self):
        # Define game state
        game_state = game.GameState()
        action = 0

        # Game Loop
        while True:
            # Select action
            action = np.zeros([self.Num_action])

            # Find legal moves
            count_move = 0
            legal_index = []
            for i in range(self.Num_spot):
                for j in range(self.Num_spot):
                    # Append legal move index into list
                    if self.gameboard[i, j] == 0:
                        legal_index.append(count_move)
                    count_move += 1

            # Randomly take action among legal actions
            if len(legal_index) > 0:
                action[random.choice(legal_index)] = 1

            # Take action and get info. for update
            self.gameboard, self.state, self.check_valid_pos, self.win_index, self.turn = game_state.step(
                action)

            # Delay for visualization
            # time.sleep(0.01)


if __name__ == '__main__':
    agent = self_demo()
    agent.main()
