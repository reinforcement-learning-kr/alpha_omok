import numpy as np
from agents import Agent


class AgentInfo:

    def __init__(self, board_size):

        self.p = np.zeros([board_size, board_size])
        self.p_size = board_size * board_size
        self.visit = np.zeros([board_size, board_size])
        self.visit_size = board_size * board_size
        self.moves = []
        self.values = []
        self.agent = Agent(board_size)

    def add_value(self, move, value):

        self.moves.append(move)

        value = (value + 1.0) / 2.0 * 100.0

        self.values.append(value)

    def clear_values(self):

        self.moves = []
        self.values = []
