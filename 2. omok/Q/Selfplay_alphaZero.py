from utils import valid_actions, check_win, render_str
from copy import deepcopy
import numpy as np
import random
import time
import tensorflow as tf

import env_small as game

import tree as MCTS_tree

class AlphaZero:
	def __init__(self):
		self.state_size, self.action_size, self.win_mark = game.Return_BoardParams()
		self.channel_size = 17

		self.learning_rate = 0.00025
		self.GPU_fraction = 0.6

		# Initialize Network
		self.input, self.is_train, self.output_policy, self.output_value = self.network()
		self.reward_loss, self.move_loss, self.train_step, self.loss = self.loss_and_train()
		self.sess = self.init_sess()

	def main(self):
		env = game.GameState('pygame')

		board_shape = [self.state_size, self.state_size]
		game_board = np.zeros(board_shape)

		replay_memory = []

		do_mcts = True
		num_mcts = 1000

		# 0: O, 1: X
		turn = 0
		turn_str = ['Black', 'White']

		# Initialize tree root id
		root_id = (0,)

		tree = {root_id: {'board': game_board,
		                  'player': turn,
		                  'child': [],
		                  'parent': None,
		                  'n': 0,
		                  'w': None,
		                  'q': None}}

		while True:
			# Select action
			action = np.zeros([self.state_size, self.state_size])

			# MCTS
			if do_mcts:
				if root_id != (0,):
				    # Delete useless tree elements
				    tree_keys = list(tree.keys())

				    for key in tree_keys:
				        if root_id != key[:len(root_id)]:
				            del tree[key]

				else:
				    # Initialize Tree
				    tree = {root_id: {'board': game_board,
				                      'player': turn,
				                      'child': [],
				                      'parent': None,
				                      'n': 0,
				                      'w': None,
				                      'q': None}}
				print('==========================')
				for i in range(num_mcts):
				    # Show progress
				    if i % (num_mcts/10) == 0:
				        print('Doing MCTS: ' + str(i) + ' / ' + str(num_mcts))

				    # step 1: selection
				    leaf_id = self.selection(tree, root_id)
				    # step 2: expansion
				    tree, child_id = self.expansion(tree, leaf_id)
				    # step 3: simulation
				    sim_result = self.simulation(tree, child_id)
				    # step 4: backup
				    tree = self.backup(tree, child_id, sim_result, root_id)

				print("\n--> " + turn_str[tree[root_id]['player']] + "'s turn <--\n")
				render_str(tree[root_id]['board'], state_size)
				q_list = {}
				actions = tree[root_id]['child']
				for i in actions:
				    q_list[root_id + (i,)] = tree[root_id + (i,)]['q']

				# Find Max Action
				max_action = max(q_list, key=q_list.get)[-1]
				max_row = int(max_action/self.state_size)
				max_col = int(max_action%self.state_size)

				print('max index: ' + '(' + str(max_row) + ' , ' + str(max_col) + ')')

				action = np.zeros([self.state_size * self.state_size])
				action[max_action] = 1

				replay_memory.append([state, action])
				################################################################################################

			# Take action and get info. for update
			game_board, state, check_valid_pos, win_index, turn, coord = env.step(action)

			# If one move is done
			if check_valid_pos:
			    last_action_idx = coord[0] * self.state_size + coord[1]

			    root_id = root_id + (last_action_idx,)

			# If game is finished
			if win_index != 0:
			    game_board = np.zeros([self.state_size, self.state_size])

			    # # Human wins!
			    # if turn == ai_turn:
			    #     ai_turn = 0
			    #     do_mcts = True
			    # else:
			    #     ai_turn = 1
			    #     do_mcts = False

			# Delay for visualization
			time.sleep(0.01)

	# -------------------------- MCTS -------------------------- #
	def selection(self, tree, root_id):
		node_id = root_id

		while True:
			num_child = len(tree[node_id]['child'])
			# Check if current node is leaf node
			if num_child == 0:
				return node_id
			else:
				max_value = -1000
				leaf_id = node_id

				policy = self.sess.run(self.output_policy, feed_dict = {self.input: tree[node_id]['board'],
																		self.is_train: False})

				for i in range(num_child):
					action = tree[leaf_id]['child'][i]
					child_id = leaf_id + (action,)
					w = tree[child_id]['w']
					n = tree[child_id]['n']
					total_n = tree[tree[child_id]['parent']]['n']

					# for unvisited child, cannot compute u value
					# so make n to be very small number
					c = np.sqrt(2)
					p = policy[action]
					q = w / n

					u = c * p * np.sqrt(np.log(total_n) / (n+1))

					if q + u > max_value:
						max_value = q + u
						node_id = child_id

	def expansion(self, tree, leaf_id):
	    leaf_state = deepcopy(tree[leaf_id]['board'])
	    is_terminal = check_win(leaf_state, self.win_mark)
	    actions = valid_actions(leaf_state)
	    # expand_thres = 1
	    #
	    # if leaf_id == (0,) or tree[leaf_id]['n'] >= expand_thres:
	    #     is_expand = True
	    # else:
	    #     is_expand = False

	    if is_terminal == 0:
	        # expansion for every possible actions
	        childs = []
	        for action in actions:
	            state = deepcopy(tree[leaf_id]['board'])
	            action_index = action[1]
	            current_player = tree[leaf_id]['player']

	            if current_player == 0:
	                next_turn = 1
	                state[action[0]] = 1
	            else:
	                next_turn = 0
	                state[action[0]] = -1

	            child_id = leaf_id + (action_index, )
	            childs.append(child_id)
	            tree[child_id] = {'board': state,
	                              'player': next_turn,
	                              'child': [],
	                              'parent': leaf_id,
	                              'n': 0,
	                              'w': 0,
	                              'q': 0}

	            tree[leaf_id]['child'].append(action_index)

	        child_id = random.sample(childs, 1)
	        return tree, child_id[0]
	    else:
	        # If leaf node is terminal state,
	        # just return MCTS tree
	        return tree, leaf_id

	def simulation(self, tree, child_id):
	    state = deepcopy(tree[child_id]['board'])
	    player = deepcopy(tree[child_id]['player'])

	    while True:
	        win = check_win(state, self.win_mark)

	        if win != 0:
	            return win
	        else:
	            actions = valid_actions(state)
	            action = random.choice(actions)
	            if player == 0:
	                player = 1
	                state[action[0]] = 1
	            else:
	                player = 0
	                state[action[0]] = -1

	def backup(self, tree, child_id, sim_result, root_id):
	    player = deepcopy(tree[root_id]['player'])
	    node_id = child_id

	    # sim_result: 1 = O win, 2 = X win, 3 = Draw
	    if sim_result == 3:
	        value = 0
	    elif sim_result - 1 == player:
	        value = 1
	    else:
	        value = -1

	    while True:
	        tree[node_id]['n'] += 1
	        tree[node_id]['w'] += value
	        tree[node_id]['q'] = tree[node_id]['w'] / tree[node_id]['n']

	        parent_id = tree[node_id]['parent']

	        if parent_id == root_id:
	            tree[parent_id]['n'] += 1
	            return tree
	        else:
	            node_id = parent_id

	def init_sess(self):
		# Initialize variables
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = self.GPU_fraction

		sess = tf.InteractiveSession(config=config)
		init = tf.global_variables_initializer()
		sess.run(init)

		return sess

	# Convolution and pooling
	def conv2d(self, x, w, stride):
		return tf.nn.conv2d(x,w,strides=[1, stride, stride, 1], padding='SAME')

	# Get Variables
	def conv_weight_variable(self, name, shape):
	    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer_conv2d())

	def weight_variable(self, name, shape):
	    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

	def bias_variable(self, name, shape):
	    return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())

	def network(self):
		# Input
		x_image = tf.placeholder(tf.float32, shape = [None,
													  self.state_size,
													  self.state_size,
													  self.channel_size])

		is_train = tf.placeholder(tf.bool)

		# Convolution variables
		w_conv1 = self.conv_weight_variable('w_conv1', [3,3,self.channel_size,256])
		b_conv1 = self.bias_variable('b_conv1',[256])

		w_conv2 = self.conv_weight_variable('w_conv2', [3,3,256,256])
		b_conv2 = self.bias_variable('b_conv2',[256])

		# Policy variables
		w_conv_policy = self.conv_weight_variable('w_conv_policy', [1, 1, 256, 2])
		b_conv_policy = self.bias_variable('b_conv_policy',[2])

		w_fc_policy = self.weight_variable('w_fc_policy', [self.action_size * 2, self.action_size])

		# Value variables
		w_conv_value = self.conv_weight_variable('w_conv_value', [1, 1, 256, 1])
		b_conv_value = self.bias_variable('b_conv_value',[1])

		w_fc_value1 = self.weight_variable('w_fc_value1', [self.action_size, 256])
		b_fc_value1 = self.bias_variable('b_fc_value1',[256])
		w_fc_value2 = self.weight_variable('w_fc_value2', [256, 1])

		# Residual block
		h_conv1 = self.conv2d(x_image, w_conv1, 1) + b_conv1
		h_BN1   = tf.layers.batch_normalization(h_conv1, training=is_train)
		h_relu1 = tf.nn.relu(h_BN1)

		h_conv2 = self.conv2d(h_relu1, w_conv2, 1) + b_conv2
		h_BN2   = tf.layers.batch_normalization(h_conv2, training=is_train)
		h_relu2 = tf.nn.relu(h_BN2)

		# Policy head
		h_conv_policy = self.conv2d(h_relu2, w_conv_policy, 1) + b_conv_policy
		h_BN_policy   = tf.layers.batch_normalization(h_conv_policy, training=is_train)
		h_relu_policy = tf.nn.relu(h_BN_policy)
		h_policy_flat = tf.reshape(h_relu_policy, [-1, self.action_size * 2])
		output_policy = tf.matmul(h_policy_flat, w_fc_policy)

		# Value head
		h_conv_value  = self.conv2d(h_relu2, w_conv_value, 1) + b_conv_value
		h_BN_value    = tf.layers.batch_normalization(h_conv_value, training=is_train)
		h_relu_value1 = tf.nn.relu(h_BN_value)
		h_value_flat  = tf.reshape(h_relu_policy, [-1, self.action_size])
		h_fc_value    = tf.matmul(h_value_flat, w_fc_value1) + b_fc_value1
		h_relu_value2 = tf.nn.relu(h_fc_value)
		output_value  = tf.tanh(tf.matmul(h_relu_value2, w_fc_value2))

		return x_image, is_train, output_policy, output_value

	def loss_and_train(self):
		# Loss function and Train
		reward_loss = tf.placeholder(tf.float32, shape = [None, 1])
		move_loss   = tf.placeholder(tf.float32, shape = [None, self.action_size]) # One hot

		loss_value = tf.square(tf.subtract(reward_loss, self.output_value))
		loss_policy =  - tf.reduce_sum(tf.multiply(move_loss, tf.log(self.output_policy)), axis = 1)

		loss = loss_value + loss_policy

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			train_step = tf.train.AdamOptimizer(learning_rate = self.learning_rate, epsilon = 1e-02).minimize(loss)

		return reward_loss, move_loss, train_step, loss

if __name__ == '__main__':
    agent = AlphaZero()
    agent.main()
