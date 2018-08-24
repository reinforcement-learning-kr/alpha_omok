# Mini Omok
'''
This is mini version of omok.
Win: Black or white stone has to be 5 in a row (horizontal, vertical, diagonal)
boardsize: 9 x 9
'''
# By KyushikMin kyushikmin@gamil.com
# http://mmc.hanyang.ac.kr

import sys

import numpy as np
import pygame
from pygame.locals import *

from utils import check_win

GAMEBOARD_SIZE = 9
WIN_STONES = 5

# Window Information
FPS = 30
WINDOW_WIDTH = 300
WINDOW_HEIGHT = 450
TOP_MARGIN = 160
MARGIN = 10
BOARD_MARGIN = 20
INPUT_CHANNEL = 3
GRID_SIZE = WINDOW_WIDTH - 2 * (BOARD_MARGIN + MARGIN)

HALF_WINDOW_WIDTH = int(WINDOW_WIDTH / 2)
HALF_WINDOW_HEIGHT = int(WINDOW_HEIGHT / 2)

# Colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 72, 72)
LIGHT_ORANGE = (198, 108, 58)
ORANGE = (180, 122, 48)
GREEN = (72, 160, 72)
BLUE = (66, 72, 200)
YELLOW = (162, 162, 42)
NAVY = (75, 0, 130)
PURPLE = (143, 0, 255)
BADUK = (220, 179, 92)


def ReturnName():
    return 'mini_omok'


def Return_Num_Action():
    return GAMEBOARD_SIZE * GAMEBOARD_SIZE


def Return_BoardParams():
    return GAMEBOARD_SIZE, GAMEBOARD_SIZE**2


class GameState:
    def __init__(self, gamemode):
        global DISPLAYSURF, BASIC_FONT, TITLE_FONT, GAMEOVER_FONT

        self.gamemode = gamemode

        if self.gamemode == 'pygame':
            pygame.init()

            DISPLAYSURF = pygame.display.set_mode(
                (WINDOW_WIDTH, WINDOW_HEIGHT))

            pygame.display.set_caption('Mini Omok')
            # pygame.display.set_icon(pygame.image.load('./Qar_Sim/icon_resize2.png'))

            BASIC_FONT = pygame.font.Font('freesansbold.ttf', 12)
            TITLE_FONT = pygame.font.Font('freesansbold.ttf', 24)
            GAMEOVER_FONT = pygame.font.Font('freesansbold.ttf', 54)

        # Set initial parameters
        self.init = False
        self.num_stones = 0

        # No stone: 0, Black stone: 1, White stone = -1
        self.gameboard = np.zeros([GAMEBOARD_SIZE, GAMEBOARD_SIZE])
        # self.state = np.zeros([GAMEBOARD_SIZE, GAMEBOARD_SIZE, INPUT_CHANNEL])
        # self.state[:, :, 16] = 1

        self.black_win = 0
        self.white_win = 0
        self.count_draw = 0

        # black turn: 0, white turn: 1
        self.turn = 0

        # List of X coordinates and Y coordinates
        self.X_coord = []
        self.Y_coord = []

        for i in range(GAMEBOARD_SIZE):
            self.X_coord.append(MARGIN + BOARD_MARGIN +
                                i * int(GRID_SIZE / (GAMEBOARD_SIZE - 1)))
            self.Y_coord.append(TOP_MARGIN + BOARD_MARGIN + i *
                                int(GRID_SIZE / (GAMEBOARD_SIZE - 1)))

    # Game loop
    def step(self, input_):
        # Initial settings
        if self.init is True:
            self.num_stones = 0

            # No stone: 0, Black stone: 1, White stone = -1
            self.gameboard = np.zeros([GAMEBOARD_SIZE, GAMEBOARD_SIZE])

            # black turn: 0, white turn: 1
            self.turn = 0

            self.init = False

        # Key settings
        mouse_pos = 0
        if np.all(input_) == 0 and self.gamemode == 'pygame':
            # If guide mode of O's turn
            for event in pygame.event.get():  # event loop
                if event.type == QUIT:
                    self.terminate()

                if pygame.mouse.get_pressed()[0]:
                    mouse_pos = pygame.mouse.get_pos()

        # get action and put stone on the board
        check_valid_pos = False
        x_index = 100
        y_index = 100

        # action = np.reshape(input_, (GAMEBOARD_SIZE, GAMEBOARD_SIZE))

        action_index = 0
        if mouse_pos != 0:
            for i in range(len(self.X_coord)):
                for j in range(len(self.Y_coord)):
                    if ((self.X_coord[i] - 15 < mouse_pos[0] < self.X_coord[i] + 15) and
                            (self.Y_coord[j] - 15 < mouse_pos[1] < self.Y_coord[j] + 15)):
                        check_valid_pos = True
                        x_index = i
                        y_index = j

                        action_index = y_index * GAMEBOARD_SIZE + x_index

                        # If selected spot is already occupied, it is not valid move!
                        if self.gameboard[y_index, x_index] == 1 or self.gameboard[y_index, x_index] == -1:
                            check_valid_pos = False

        # If self mode and MCTS works
        if np.any(input_) != 0:
            action_index = np.argmax(input_)
            y_index = int(action_index / GAMEBOARD_SIZE)
            x_index = action_index % GAMEBOARD_SIZE
            check_valid_pos = True

            # If selected spot is already occupied, it is not valid move!
            if self.gameboard[y_index, x_index] == 1 or self.gameboard[y_index, x_index] == -1:
                check_valid_pos = False

        # Change the gameboard according to the stone's index
        if np.any(input_) != 0:
            # update state
            # self.state = update_state(self.state, self.turn, x_index, y_index)

            if self.turn == 0:
                self.gameboard[y_index, x_index] = 1
                self.turn = 1
                self.num_stones += 1
            else:
                self.gameboard[y_index, x_index] = -1
                self.turn = 0
                self.num_stones += 1

        if self.gamemode == 'pygame':
            # Fill background color
            DISPLAYSURF.fill(BLACK)

            # Draw board
            self.draw_main_board()

            # Display Information
            self.title_msg()
            self.rule_msg()
            self.score_msg()

            # Display who's turn
            self.turn_msg()

            pygame.display.update()

        # Check_win 0: playing, 1: black win, 2: white win, 3: draw
        win_index = check_win(self.gameboard, WIN_STONES)
        self.display_win(win_index)

        return self.gameboard, check_valid_pos, win_index, self.turn, action_index

    # Exit the game
    def terminate(self):
        pygame.quit()
        sys.exit()

    # Draw main board
    def draw_main_board(self):
        # Main board size = 400 x 400
        # Game board size = 320 x 320
        mainboard_rect = pygame.Rect(MARGIN, TOP_MARGIN, WINDOW_WIDTH -
                                     2 * MARGIN, WINDOW_WIDTH - 2 * MARGIN)
        pygame.draw.rect(DISPLAYSURF, BADUK, mainboard_rect)

        # Horizontal Lines
        for i in range(GAMEBOARD_SIZE):
            pygame.draw.line(DISPLAYSURF, BLACK, (MARGIN + BOARD_MARGIN, TOP_MARGIN + BOARD_MARGIN + i * int(GRID_SIZE / (GAMEBOARD_SIZE - 1))),
                             (WINDOW_WIDTH - (MARGIN + BOARD_MARGIN), TOP_MARGIN + BOARD_MARGIN + i * int(GRID_SIZE / (GAMEBOARD_SIZE - 1))), 1)

        # Vertical Lines
        for i in range(GAMEBOARD_SIZE):
            pygame.draw.line(DISPLAYSURF, BLACK, (MARGIN + BOARD_MARGIN + i * int(GRID_SIZE / (GAMEBOARD_SIZE - 1)), TOP_MARGIN + BOARD_MARGIN),
                             (MARGIN + BOARD_MARGIN + i * int(GRID_SIZE / (GAMEBOARD_SIZE - 1)), TOP_MARGIN + BOARD_MARGIN + GRID_SIZE), 1)

        # Draw center circle
        pygame.draw.circle(DISPLAYSURF, BLACK, (MARGIN + BOARD_MARGIN + 4 * int(GRID_SIZE / (
            GAMEBOARD_SIZE - 1)), TOP_MARGIN + BOARD_MARGIN + 4 * int(GRID_SIZE / (GAMEBOARD_SIZE - 1))), 5, 0)

        # Draw stones
        for i in range(self.gameboard.shape[0]):
            for j in range(self.gameboard.shape[1]):
                if self.gameboard[i, j] == 1:
                    pygame.draw.circle(DISPLAYSURF, BLACK,
                                       (self.X_coord[j], self.Y_coord[i]), 12, 0)

                if self.gameboard[i, j] == -1:
                    pygame.draw.circle(DISPLAYSURF, WHITE,
                                       (self.X_coord[j], self.Y_coord[i]), 12, 0)

    # Display title
    def title_msg(self):
        titleSurf = TITLE_FONT.render('Mini Omok', True, WHITE)
        titleRect = titleSurf.get_rect()
        titleRect.topleft = (MARGIN, 10)
        DISPLAYSURF.blit(titleSurf, titleRect)

    # Display rule
    def rule_msg(self):
        ruleSurf1 = BASIC_FONT.render(
            'Win: Stones has to be 5 in a row', True, WHITE)
        ruleRect1 = ruleSurf1.get_rect()
        ruleRect1.topleft = (MARGIN, 50)
        DISPLAYSURF.blit(ruleSurf1, ruleRect1)

        ruleSurf2 = BASIC_FONT.render(
            '(horizontal, vertical, diagonal)', True, WHITE)
        ruleRect2 = ruleSurf1.get_rect()
        ruleRect2.topleft = (MARGIN + 35, 70)
        DISPLAYSURF.blit(ruleSurf2, ruleRect2)

    # Display scores
    def score_msg(self):
        scoreSurf1 = BASIC_FONT.render('Score: ', True, WHITE)
        scoreRect1 = scoreSurf1.get_rect()
        scoreRect1.topleft = (MARGIN, 105)
        DISPLAYSURF.blit(scoreSurf1, scoreRect1)

        scoreSurf2 = BASIC_FONT.render(
            'Black = ' + str(self.black_win) + '  vs  ', True, WHITE)
        scoreRect2 = scoreSurf2.get_rect()
        scoreRect2.topleft = (scoreRect1.midright[0], 105)
        DISPLAYSURF.blit(scoreSurf2, scoreRect2)

        scoreSurf3 = BASIC_FONT.render(
            'White = ' + str(self.white_win) + '  vs  ', True, WHITE)
        scoreRect3 = scoreSurf3.get_rect()
        scoreRect3.topleft = (scoreRect2.midright[0], 105)
        DISPLAYSURF.blit(scoreSurf3, scoreRect3)

        scoreSurf4 = BASIC_FONT.render(
            'Draw = ' + str(self.count_draw), True, WHITE)
        scoreRect4 = scoreSurf4.get_rect()
        scoreRect4.topleft = (scoreRect3.midright[0], 105)
        DISPLAYSURF.blit(scoreSurf4, scoreRect4)

    # Display turn
    def turn_msg(self):
        if self.turn == 0:
            turnSurf = BASIC_FONT.render("Black's Turn!", True, WHITE)
            turnRect = turnSurf.get_rect()
            turnRect.topleft = (MARGIN, 135)
            DISPLAYSURF.blit(turnSurf, turnRect)
        else:
            turnSurf = BASIC_FONT.render("White's Turn!", True, WHITE)
            turnRect = turnSurf.get_rect()
            turnRect.topleft = (WINDOW_WIDTH - 110, 135)
            DISPLAYSURF.blit(turnSurf, turnRect)

    # Display Win
    def display_win(self, win_index):
        # Black Win
        if win_index == 1:
            self.black_win += 1
            self.init = True

        # White Win
        elif win_index == 2:
            self.white_win += 1
            self.init = True

        # Draw
        elif win_index == 3:
            self.count_draw += 1
            self.init = True

        else:
            self.init = False
