import numpy as np

class GameInfo:

    def __init__(self):
        
        self.game_board = None
        self.win_index = 0
        self.curr_turn = 0
        self.gmae_board_message = ''