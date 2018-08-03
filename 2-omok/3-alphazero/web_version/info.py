import numpy as np


class GameInfo:

    def __init__(self, board_size):

        self.game_board = np.zeros([board_size, board_size])
        self.win_index = 0
        self.curr_turn = 0
        self.player_message = ''
        self.enemy_message = ''


class AgentInfo:

    def __init__(self, board_size):

        self.pi = np.zeros([board_size, board_size])
        self.pi_size = board_size
        self.visit = np.zeros([board_size, board_size])
        self.visit_size = board_size
        self.moves = []
        self.values = []
        self.message = ''

    def add_value(self, move, value):

        self.moves.append(move)

        value = (value + 1.0) / 2.0 * 100.0

        self.values.append(value)

    def clear_values(self):

        self.moves = []
        self.values = []
