import numpy as np


class Node(object):

    def __init__(self, state):
        self.state = state
        self.player_turn = state.player_turn
        self.id = state.id
        self.edges = []

    def is_leaf(self):
        if len(self.edges) > 0:
            return False
        else:
            return True


class Edge(object):

    def __init__(self, parent_node, child_node, p_theta, action):
        self.id = parent_node.state.id + '|' + child_node.state.id
        self.parent_node = parent_node
        self.child_node = child_node
        self.player_turn = parent_node.state.player_turn
        self.action = action

        self.stat = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': p_theta,
        }


class MCTS(object):

    def __init__(self, root):
        self.root = root
        self.tree = {}

        # hyperparameter
        self.c_puct = 5
        self.epsilon = 0.25
        self.alpha = 0.7

        self.expand(root)

    def __len__(self):
        return len(self.tree)

    def select(self):

        trajectory = []
        current_node = self.root

        done = 0
        value = 0

        while not current_node.is_leaf():

            puct = -np.inf

            if current_node == self.root:
                epsilon = self.epsilon
                noise = np.random.dirichlet(self.alpha * len(current_node.edges))
            else:
                epsilon = 0
                noise = [0] * len(current_node.edges)

            total_visit = 0
            for action, edge in current_node.edges:
                total_visit += edge.stat['N']

            for i, (action, edge) in enumerate(current_node.edges):

                U = self.c_puct * \
                    ((1 - epsilon) * edge.stat['P'] + epsilon * noise[i]) * \
                    np.sqrt(total_visit) / (1 + edge.stat['N'])

                Q = edge.stat['Q']

                if Q + U > puct:
                    puct = Q + U
                    simulation_action = action
                    simulation_edge = edge

            # the value of the new_state from the POV of the new player_turn
            new_state, value, done = current_node.state.action(simulation_action)
            current_node = simulation_edge.child_node
            trajectory.append(simulation_edge)

        return current_node, value, done, trajectory

    def backup(self, leaf, value, trajectory):

        current_player = leaf.state.player_turn

        for edge in trajectory:
            player_turn = edge.player_turn
            if player_turn == current_player:
                correction = 1
            else:
                correction = -1

            edge.stat['N'] = edge.stat['N'] + 1
            edge.stat['W'] = edge.stat['W'] + value * correction
            edge.stat['Q'] = edge.stat['W'] / edge.stat['N']

    def expand(self, node):
        self.tree[node.id] = node
