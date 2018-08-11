import numpy as np


class GameInfo:

    def __init__(self, board_size):

        self.game_board = np.zeros([board_size, board_size])
        self.win_index = 0
        self.curr_turn = 0
        self.message = '오목'