# TicTacToe Versus version

# Import modules
import pygame
import random
import numpy as np
import time
import copy

import tictactoe_class as game

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
		self.check_valid_pos = 0
		self.win_index = 0

		####################### Parameters for MCTS (Monte Carlo Tree Search) #######################
		# Start mcts or not
		self.do_mcts = False

		# Number of iteration for each move
		self.num_MCTS_iteration = 1000
		#############################################################################################

		self.GAMEBOARD_SIZE, self.WIN_MARK = game.Return_BoardParams();

	def main(self):
		# Define game state
		game_state = game.GameState()

		# Game Loop
		while True:
			# Select action
			action = 0

			############################################ MCTS ############################################
			if self.do_mcts:
				start_time = time.time()
				# Initialize Tree
				MCTS_node = {(0,): {'state': self.gameboard, 'player': self.turn, 'child': [], 'parent': None, 'total_n': 0}}
				MCTS_edge = {}

				count = 0
				for i in range(self.num_MCTS_iteration):
					leafnode_id = self.MCTS_search(MCTS_node, MCTS_edge)
					MCTS_node, MCTS_edge, update_node_id = self.MCTS_expand(MCTS_node, MCTS_edge, leafnode_id)
					# sim_result: 1 = O win, 2 = X win, 3 = Draw
					sim_result = self.MCTS_simulation(MCTS_node, MCTS_edge, update_node_id)
					MCTS_node, MCTS_edge = self.MCTS_backup(MCTS_node, MCTS_edge, update_node_id, sim_result)
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

				self.do_mcts = False
				action = np.zeros([self.GAMEBOARD_SIZE * self.GAMEBOARD_SIZE])
				action[max_action] = 1

				print('MCTS Calculation time: ' + str(time.time() - start_time))

			################################################################################################

			# Take action and get info. for update
			self.gameboard, self.check_valid_pos, self.win_index, self.turn = game_state.frame_step(action)

			# If one move is done
			if self.turn == 1:
				self.do_mcts = True

			# If game is finished
			if self.win_index != 0:
				self.gameboard = np.zeros([self.Num_spot, self.Num_spot])

			# Delay for visualization
			time.sleep(0.01)

	# Search (MCTS)
	def MCTS_search(self, MCTS_node, MCTS_edge):
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

	def MCTS_expand(self,MCTS_node, MCTS_edge, leafnode_id):
		#Find legal move
		current_board = copy.deepcopy(MCTS_node[leafnode_id]['state'])
		is_terminal = self.check_win(current_board, np.count_nonzero(current_board))
		legal_moves = self.find_legal_moves(current_board)
		expand_count = 5

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


	def MCTS_simulation(self,MCTS_node, MCTS_edge, update_node_id):
	    current_board  = copy.deepcopy(MCTS_node[update_node_id]['state'])
	    current_player = copy.deepcopy(MCTS_node[update_node_id]['player'])
	    while True:
	        if self.check_win(current_board, np.count_nonzero(current_board)) != 0:
	            return self.check_win(current_board, np.count_nonzero(current_board))
	        else:
	            legal_moves = self.find_legal_moves(current_board)

	            chosen_move = random.choice(legal_moves)
	            chosen_coord = chosen_move[0]
	            chosen_index = chosen_move[1]

	            if current_player == 0:
	                current_player = 1
	                current_board[chosen_coord[0]][chosen_coord[1]] = 1
	            else:
	                current_player = 0
	                current_board[chosen_coord[0]][chosen_coord[1]] = -1

	def MCTS_backup(self,MCTS_node, MCTS_edge, update_node_id, sim_result):
		current_player = copy.deepcopy(MCTS_node[(0,)]['player'])
		current_id = update_node_id

		if sim_result == 3:
		    value = 0.7
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

	def find_legal_moves(self, gameboard):
	    legal_moves = []
	    count_moves = 0
	    for i in range(self.GAMEBOARD_SIZE ):
	        for j in range(self.GAMEBOARD_SIZE ):
	            if gameboard[i][j] == 0:
	                legal_moves.append([(i,j), count_moves])
	            count_moves += 1
	    return legal_moves

	# Check win
	def check_win(self, gameboard, num_mark):
		# Check four stones in a row (Horizontal)
		for row in range(self.GAMEBOARD_SIZE ):
		    for col in range(self.GAMEBOARD_SIZE  - self.WIN_MARK + 1):
		        # Black win!
		        if np.sum(gameboard[row, col:col + self.WIN_MARK]) == self.WIN_MARK:
		            return 1
		        # White win!
		        if np.sum(gameboard[row, col:col + self.WIN_MARK]) == -self.WIN_MARK:
		            return 2

		# Check four stones in a colum (Vertical)
		for row in range(self.GAMEBOARD_SIZE  - self.WIN_MARK + 1):
		    for col in range(self.GAMEBOARD_SIZE ):
		        # Black win!
		        if np.sum(gameboard[row : row + self.WIN_MARK, col]) == self.WIN_MARK:
		            return 1
		        # White win!
		        if np.sum(gameboard[row : row + self.WIN_MARK, col]) == -self.WIN_MARK:
		            return 2

		# Check four stones in diagonal (Diagonal)
		for row in range(self.GAMEBOARD_SIZE  - self.WIN_MARK + 1):
		    for col in range(self.GAMEBOARD_SIZE  - self.WIN_MARK + 1):
		        count_sum = 0
		        for i in range(self.WIN_MARK):
		            if gameboard[row + i, col + i] == 1:
		                count_sum += 1
		            if gameboard[row + i, col + i] == -1:
		                count_sum -= 1

		        # Black Win!
		        if count_sum == self.WIN_MARK:
		            return 1

		        # White WIN!
		        if count_sum == -self.WIN_MARK:
		            return 2

		for row in range(self.WIN_MARK-1, self.GAMEBOARD_SIZE ):
		    for col in range(self.GAMEBOARD_SIZE  - self.WIN_MARK + 1):
		        count_sum = 0
		        for i in range(self.WIN_MARK):
		            if gameboard[row - i, col + i] == 1:
		                count_sum += 1
		            if gameboard[row - i, col + i] == -1:
		                count_sum -= 1

		        # Black Win!
		        if count_sum == self.WIN_MARK:
		            return 1

		        # White WIN!
		        if count_sum == -self.WIN_MARK:
		            return 2

		# Draw (board is full)
		if num_mark == self.GAMEBOARD_SIZE  * self.GAMEBOARD_SIZE :
		    return 3
			
		# If No winner or no draw
		return 0

if __name__ == '__main__':
	agent = self_demo()
	agent.main()
