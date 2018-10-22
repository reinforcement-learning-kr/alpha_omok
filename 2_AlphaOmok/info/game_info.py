import numpy as np


class GameInfo:

    def __init__(self, board_size):

        self.game_board = np.zeros([board_size, board_size])
        self.win_index = 0
        self.curr_turn = 0
        self.enemy_turn = 0
        self.action_index = -1
        self.message = '오목'
        self.player_agent_name = ''
        self.enemy_agent_name = ''
        self.enemy_action_index = -1
        self.game_status = 0
