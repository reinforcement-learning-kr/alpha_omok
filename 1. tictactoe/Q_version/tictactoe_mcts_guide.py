# Mini Omok
'''
This is tictactoe with MCTS (Monte Carlo Tree Search).
Win: O or X has to be 3 in a row (Horizontal or Vertical or Diagonal)
boardsize: 3 x 3
'''
# By KyushikMin kyushikmin@gamil.com
# http://mmc.hanyang.ac.kr

import random, sys, time, math, pygame
from pygame.locals import *
import numpy as np
import copy

# Window Information
FPS = 30
WINDOW_WIDTH = 340
WINDOW_HEIGHT = 480
TOP_MARGIN = 160
MARGIN = 20
GAMEBOARD_SIZE = 3
WIN_MARK = 3
GRID_SIZE = WINDOW_WIDTH - 2 * (MARGIN)

HALF_WINDOW_WIDTH = int(WINDOW_WIDTH / 2)
HALF_WINDOW_HEIGHT = int(WINDOW_HEIGHT / 2)

# Colors
#				 R    G    B
WHITE        = (255, 255, 255)
BLACK		 = (  0,   0,   0)
RED 		 = (200,  72,  72)
LIGHT_ORANGE = (198, 108,  58)
ORANGE       = (180, 122,  48)
GREEN		 = ( 72, 160,  72)
BLUE 		 = ( 66,  72, 200)
YELLOW 		 = (162, 162,  42)
NAVY         = ( 75,   0, 130)
PURPLE       = (143,   0, 255)
BADUK        = (220, 179,  92)

def main():
    global FPS_CLOCK, DISPLAYSURF, BASIC_FONT, TITLE_FONT, GAMEOVER_FONT

    pygame.init()
    FPS_CLOCK = pygame.time.Clock()

    DISPLAYSURF = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

    pygame.display.set_caption('TicTacToe')

    BASIC_FONT = pygame.font.Font('freesansbold.ttf', 16)
    TITLE_FONT = pygame.font.Font('freesansbold.ttf', 24)
    GAMEOVER_FONT = pygame.font.Font('freesansbold.ttf', 48)

    # Set initial parameters
    init = False
    num_mark = 0

    # No stone: 0, Black stone: 1, White stone = -1
    gameboard = np.zeros([GAMEBOARD_SIZE, GAMEBOARD_SIZE])

    x_win = 0
    o_win = 0
    count_draw = 0

    # black turn: 0, white turn: 1
    turn = 0

    # List of X coordinates and Y coordinates
    X_coord = []
    Y_coord = []

    ####################### Parameters for MCTS (Monte Carlo Tree Search) #######################
    # Start mcts or not
    do_mcts = True

    # Number of iteration for each move
    num_MCTS_iteration = 1000
    #############################################################################################

    for i in range(GAMEBOARD_SIZE):
        X_coord.append(MARGIN + i * int(GRID_SIZE/(GAMEBOARD_SIZE)) + int(GRID_SIZE/(GAMEBOARD_SIZE * 2)))
        Y_coord.append(TOP_MARGIN + i * int(GRID_SIZE/(GAMEBOARD_SIZE)) + int(GRID_SIZE/(GAMEBOARD_SIZE * 2)))

    while True: # Game loop
        # Initial settings
        if init == True:
            num_mark = 0

            # No mark: 0, o: 1, x = -1
            gameboard = np.zeros([GAMEBOARD_SIZE, GAMEBOARD_SIZE])

            # o turn: 0, z turn: 1
            turn = 0

            # Start mcts or not
            do_mcts = True

            init = False


        ############################################ MCTS ############################################
        if do_mcts:
            start_time = time.time()
            # Initialize Tree
            MCTS_node = {(0,): {'state': gameboard, 'player': turn, 'child': [], 'parent': None, 'total_n': 0}}
            MCTS_edge = {}

            count = 0
            for i in range(num_MCTS_iteration):
                leafnode_id = MCTS_search(MCTS_node, MCTS_edge)
                MCTS_node, MCTS_edge, update_node_id = MCTS_expand(MCTS_node, MCTS_edge, leafnode_id)
                # sim_result: 1 = O win, 2 = X win, 3 = Draw
                sim_result = MCTS_simulation(MCTS_node, MCTS_edge, update_node_id)
                MCTS_node, MCTS_edge =  MCTS_backup(MCTS_node, MCTS_edge, update_node_id, sim_result)
                count += 1

            print('=================================')
            for i in range(3):
                print(MCTS_node[(0,)]['state'][i,:])

            print('======================== Root Node ========================')
            print(MCTS_node[(0,)])

            print('======================== Edge ========================')
            Q_list = {}
            for i in MCTS_node[(0,)]['child']:
                print('Edge_id: ' + str([0,i]))
                print('Edge Value: ' + str(MCTS_edge[(0,i)]))
                Q_list[(0,i)] = MCTS_edge[(0,i)]['Q']

            # Find Max Action
            max_action = max(Q_list, key = Q_list.get)[1]
            print('\nMax Action: ' + str(max_action + 1))
            do_mcts = False
            print('MCTS Calculation time: ' + str(time.time() - start_time))

        ################################################################################################

        # Key settings
        mouse_pos = 0
        for event in pygame.event.get(): # event loop
            if event.type == QUIT:
                terminate()

            if pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()

        # Check mouse position and count
        check_valid_pos = False
        x_index = -1
        y_index = -1

        if mouse_pos != 0:
            for i in range(len(X_coord)):
                for j in range(len(Y_coord)):
                    if (X_coord[i] - 30 < mouse_pos[0] < X_coord[i] + 30) and (Y_coord[j] - 30 < mouse_pos[1] < Y_coord[j] + 30):
                        check_valid_pos = True
                        x_index = i
                        y_index = j

                        # If selected spot is already occupied, it is not valid move!
                        if gameboard[y_index, x_index] == 1 or gameboard[y_index, x_index] == -1:
                            check_valid_pos = False

        # Change the gameboard according to the stone's index
        if check_valid_pos:
            if turn == 0:
                gameboard[y_index, x_index] = 1
                turn = 1
                num_mark += 1
                do_mcts = True
            else:
                gameboard[y_index, x_index] = -1
                turn = 0
                num_mark += 1
                do_mcts = True

        # Fill background color
        DISPLAYSURF.fill(BLACK)

        # Draw board
        draw_main_board(gameboard, X_coord, Y_coord)

        # Display Information
        title_msg()
        rule_msg()
        score_msg(o_win, x_win, count_draw)

        # Display who's turn
        turn_msg(turn)

        pygame.display.update()
        FPS_CLOCK.tick(FPS)

        # Check_win 0: playing, 1: black win, 2: white win, 3: draw
        win_index = check_win(gameboard, num_mark)
        init, o_win, x_win, count_draw = display_win(win_index, o_win, x_win, count_draw)

# Search (MCTS)
def MCTS_search(MCTS_node, MCTS_edge):
    node_id = (0,)
    # Loop until finding leaf node
    while True:
        # Check if current node is leaf node
        if len(MCTS_node[node_id]['child']) == 0:
            # Leaf node!
            return node_id
        else:
            Max_QU = -100
            parent_id = node_id
            for i in range(len(MCTS_node[node_id]['child'])):
                id_temp = parent_id + (MCTS_node[parent_id]['child'][i],)
                current_w = MCTS_edge[id_temp]['W']
                current_n = copy.deepcopy(MCTS_edge[id_temp]['N'])
                parent_n  = MCTS_node[MCTS_edge[id_temp]['parent_node']]['total_n']

                if current_n == 0:
                    current_n = 0.000001

                Q = current_w / current_n
                U = 10 * np.sqrt(2 * np.log(parent_n) / current_n)

                if Q+U > Max_QU:
                    Max_QU = Q+U
                    node_id = id_temp

def MCTS_expand(MCTS_node, MCTS_edge, leafnode_id):
    #Find legal move
    current_board = copy.deepcopy(MCTS_node[leafnode_id]['state'])
    is_terminal = check_win(current_board, np.count_nonzero(current_board))
    legal_moves = find_legal_moves(current_board)
    expand_count = 30

    if leafnode_id == (0,) or MCTS_node[leafnode_id]['total_n'] > expand_count:
        is_expand = True
    else:
        is_expand = False

    if len(legal_moves) > 0 and is_terminal == 0 and is_expand:
        for legal_move in legal_moves:
            # Initialize current board at every legal move
            current_board = copy.deepcopy(MCTS_node[leafnode_id]['state'])

            chosen_coord = legal_move[0]
            chosen_index = legal_move[1]

            current_player = MCTS_node[leafnode_id]['player']

            if current_player == 0:
                next_turn = 1
                current_board[chosen_coord[0]][chosen_coord[1]] = 1
            else:
                next_turn = 0
                current_board[chosen_coord[0]][chosen_coord[1]] = -1

            child_node_id = leafnode_id + (chosen_index,)
            MCTS_node[child_node_id] = {'state': current_board,
                                        'player': next_turn,
                                        'child': [],
                                        'parent': leafnode_id,
                                        'total_n': 0}

            MCTS_edge[child_node_id] = {'N': 0, 'W': 0, 'Q': 0, 'parent_node': leafnode_id}

            MCTS_node[leafnode_id]['child'].append(chosen_index)

        return MCTS_node, MCTS_edge, child_node_id
    else:
        # If leaf node is terminal state, just return MCTS tree and True
        return MCTS_node, MCTS_edge, leafnode_id


def MCTS_simulation(MCTS_node, MCTS_edge, update_node_id):
    current_board  = copy.deepcopy(MCTS_node[update_node_id]['state'])
    current_player = copy.deepcopy(MCTS_node[update_node_id]['player'])
    while True:
        if check_win(current_board, np.count_nonzero(current_board)) != 0:
            return check_win(current_board, np.count_nonzero(current_board))
        else:
            legal_moves = find_legal_moves(current_board)

            chosen_move = random.choice(legal_moves)
            chosen_coord = chosen_move[0]
            chosen_index = chosen_move[1]

            if current_player == 0:
                current_player = 1
                current_board[chosen_coord[0]][chosen_coord[1]] = 1
            else:
                current_player = 0
                current_board[chosen_coord[0]][chosen_coord[1]] = -1

def MCTS_backup(MCTS_node, MCTS_edge, update_node_id, sim_result):
    current_player = copy.deepcopy(MCTS_node[(0,)]['player'])
    current_id = update_node_id

    # print(MCTS_tree)
    # print(update_node_id)
    # print('--------------------------------')

    if sim_result == 3:
        value = 0
    elif sim_result-1 == current_player:
        value = 1
    else:
        value = -1

    while True:
        MCTS_edge[current_id]['N'] += 1
        MCTS_edge[current_id]['W'] += value
        MCTS_edge[current_id]['Q'] = MCTS_edge[current_id]['W'] / MCTS_edge[current_id]['N']
        MCTS_node[MCTS_edge[current_id]['parent_node']]['total_n'] += 1

        if MCTS_node[current_id]['parent'] == (0,):
            return MCTS_node, MCTS_edge
        else:
            current_id = MCTS_node[current_id]['parent']


def find_legal_moves(gameboard):
    legal_moves = []
    count_moves = 0
    for i in range(GAMEBOARD_SIZE):
        for j in range(GAMEBOARD_SIZE):
            if gameboard[i][j] == 0:
                legal_moves.append([(i,j), count_moves])
            count_moves += 1
    return legal_moves


# Exit the game
def terminate():
	pygame.quit()
	sys.exit()

# Draw main board
def draw_main_board(gameboard, X_coord, Y_coord):
    # Main board size = 400 x 400
    # Game board size = 320 x 320
    # mainboard_rect = pygame.Rect(MARGIN, TOP_MARGIN, WINDOW_WIDTH - 2 * MARGIN, WINDOW_WIDTH - 2 * MARGIN)
    # pygame.draw.rect(DISPLAYSURF, BADUK, mainboard_rect)

    # Horizontal Lines
    for i in range(GAMEBOARD_SIZE+1):
        pygame.draw.line(DISPLAYSURF, WHITE, (MARGIN, TOP_MARGIN + i * int(GRID_SIZE/(GAMEBOARD_SIZE))), (WINDOW_WIDTH - (MARGIN), TOP_MARGIN + i * int(GRID_SIZE/(GAMEBOARD_SIZE))), 1)

    # Vertical Lines
    for i in range(GAMEBOARD_SIZE+1):
        pygame.draw.line(DISPLAYSURF, WHITE, (MARGIN + i * int(GRID_SIZE/(GAMEBOARD_SIZE)), TOP_MARGIN), (MARGIN + i * int(GRID_SIZE/(GAMEBOARD_SIZE)), TOP_MARGIN + GRID_SIZE), 1)

    # Draw center circle
    pygame.draw.circle(DISPLAYSURF, WHITE, (MARGIN + 4 * int(GRID_SIZE/(GAMEBOARD_SIZE)), TOP_MARGIN + 4 * int(GRID_SIZE/(GAMEBOARD_SIZE))), 5, 0)

    # Draw marks
    for i in range(gameboard.shape[0]):
        for j in range(gameboard.shape[1]):
            if gameboard[i,j] == 1:
                pygame.draw.circle(DISPLAYSURF, WHITE, (X_coord[j], Y_coord[i]), 30, 0)

            if gameboard[i,j] == -1:
                pygame.draw.line(DISPLAYSURF, WHITE, (X_coord[j] - 30, Y_coord[i] - 30), (X_coord[j] + 30, Y_coord[i] + 30), 10)
                pygame.draw.line(DISPLAYSURF, WHITE, (X_coord[j] - 30, Y_coord[i] + 30), (X_coord[j] + 30, Y_coord[i] - 30), 10)

# Display title
def title_msg():
	titleSurf = TITLE_FONT.render('TicTacToe', True, WHITE)
	titleRect = titleSurf.get_rect()
	titleRect.topleft = (MARGIN, 10)
	DISPLAYSURF.blit(titleSurf, titleRect)

# Display rule
def rule_msg():
	ruleSurf1 = BASIC_FONT.render('Win: O or X mark has to be 3 in a row', True, WHITE)
	ruleRect1 = ruleSurf1.get_rect()
	ruleRect1.topleft = (MARGIN, 50)
	DISPLAYSURF.blit(ruleSurf1, ruleRect1)

	ruleSurf2 = BASIC_FONT.render('(horizontal, vertical, diagonal)', True, WHITE)
	ruleRect2 = ruleSurf1.get_rect()
	ruleRect2.topleft = (MARGIN, 70)
	DISPLAYSURF.blit(ruleSurf2, ruleRect2)

# Display scores
def score_msg(o_win, x_win, count_draw):
    scoreSurf1 = BASIC_FONT.render('Score: ', True, WHITE)
    scoreRect1 = scoreSurf1.get_rect()
    scoreRect1.topleft = (MARGIN, 105)
    DISPLAYSURF.blit(scoreSurf1, scoreRect1)

    scoreSurf2 = BASIC_FONT.render('O = ' + str(o_win) + '  vs  ', True, WHITE)
    scoreRect2 = scoreSurf2.get_rect()
    scoreRect2.topleft = (scoreRect1.midright[0], 105)
    DISPLAYSURF.blit(scoreSurf2, scoreRect2)

    scoreSurf3 = BASIC_FONT.render('X = ' + str(x_win) + '  vs  ', True, WHITE)
    scoreRect3 = scoreSurf3.get_rect()
    scoreRect3.topleft = (scoreRect2.midright[0], 105)
    DISPLAYSURF.blit(scoreSurf3, scoreRect3)

    scoreSurf4 = BASIC_FONT.render('Draw = ' + str(count_draw), True, WHITE)
    scoreRect4 = scoreSurf4.get_rect()
    scoreRect4.topleft = (scoreRect3.midright[0], 105)
    DISPLAYSURF.blit(scoreSurf4, scoreRect4)

# Display turn
def turn_msg(turn):
    if turn == 0:
        turnSurf = BASIC_FONT.render("O's Turn!", True, WHITE)
        turnRect = turnSurf.get_rect()
        turnRect.topleft = (MARGIN, 135)
        DISPLAYSURF.blit(turnSurf, turnRect)
    else:
        turnSurf = BASIC_FONT.render("X's Turn!", True, WHITE)
        turnRect = turnSurf.get_rect()
        turnRect.topleft = (WINDOW_WIDTH - 75, 135)
        DISPLAYSURF.blit(turnSurf, turnRect)

# Check win
def check_win(gameboard, num_mark):
    # Check four stones in a row (Horizontal)
    for row in range(GAMEBOARD_SIZE):
        for col in range(GAMEBOARD_SIZE - WIN_MARK + 1):
            # Black win!
            if np.sum(gameboard[row, col:col + WIN_MARK]) == WIN_MARK:
                return 1
            # White win!
            if np.sum(gameboard[row, col:col + WIN_MARK]) == -WIN_MARK:
                return 2

    # Check four stones in a colum (Vertical)
    for row in range(GAMEBOARD_SIZE - WIN_MARK + 1):
        for col in range(GAMEBOARD_SIZE):
            # Black win!
            if np.sum(gameboard[row : row + WIN_MARK, col]) == WIN_MARK:
                return 1
            # White win!
            if np.sum(gameboard[row : row + WIN_MARK, col]) == -WIN_MARK:
                return 2

    # Check four stones in diagonal (Diagonal)
    for row in range(GAMEBOARD_SIZE - WIN_MARK + 1):
        for col in range(GAMEBOARD_SIZE - WIN_MARK + 1):
            count_sum = 0
            for i in range(WIN_MARK):
                if gameboard[row + i, col + i] == 1:
                    count_sum += 1
                if gameboard[row + i, col + i] == -1:
                    count_sum -= 1

            # Black Win!
            if count_sum == WIN_MARK:
                return 1

            # White WIN!
            if count_sum == -WIN_MARK:
                return 2

    for row in range(WIN_MARK-1, GAMEBOARD_SIZE):
        for col in range(GAMEBOARD_SIZE - WIN_MARK + 1):
            count_sum = 0
            for i in range(WIN_MARK):
                if gameboard[row - i, col + i] == 1:
                    count_sum += 1
                if gameboard[row - i, col + i] == -1:
                    count_sum -= 1

            # Black Win!
            if count_sum == WIN_MARK:
                return 1

            # White WIN!
            if count_sum == -WIN_MARK:
                return 2

    # Draw (board is full)
    if num_mark == GAMEBOARD_SIZE * GAMEBOARD_SIZE:
        return 3

    return 0

# Display Win
def display_win(win_index, o_win, x_win, count_draw):
    wait_time = 3
    # Black Win
    if win_index == 1:
        # Fill background color
        DISPLAYSURF.fill(WHITE)

        winSurf = GAMEOVER_FONT.render("O Win!", True, BLACK)
        winRect = winSurf.get_rect()
        winRect.midtop = (WINDOW_WIDTH/2, WINDOW_HEIGHT/2 - 50)
        DISPLAYSURF.blit(winSurf, winRect)
        pygame.display.update()
        time.sleep(wait_time)

        o_win += 1

        return True, o_win, x_win, count_draw

    # White Win
    if win_index == 2:
        # Fill background color
        DISPLAYSURF.fill(BLACK)

        winSurf = GAMEOVER_FONT.render("X Win!", True, WHITE)
        winRect = winSurf.get_rect()
        winRect.midtop = (WINDOW_WIDTH/2, WINDOW_HEIGHT/2 - 50)
        DISPLAYSURF.blit(winSurf, winRect)
        pygame.display.update()
        time.sleep(wait_time)

        x_win += 1

        return True, o_win, x_win, count_draw

    # Draw
    if win_index == 3:
        # Fill background color
        DISPLAYSURF.fill(WHITE)

        winSurf = GAMEOVER_FONT.render("DRAW!", True, BLACK)
        winRect = winSurf.get_rect()
        winRect.midtop = (WINDOW_WIDTH/2, WINDOW_HEIGHT/2 - 50)
        DISPLAYSURF.blit(winSurf, winRect)
        pygame.display.update()
        time.sleep(wait_time)

        count_draw += 1

        return True, o_win, x_win, count_draw

    return False, o_win, x_win, count_draw

if __name__ == '__main__':
	main()
