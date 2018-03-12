# my modules
from env import env_small as env
from utils import valid_actions
from neural_net.resnet_10block import PolicyValueNet
# built-in
from collections import defaultdict, deque
# third-party
import numpy as np


class Node:
    def __init__(self):
        self.a = defaultdict(Edge)
        self.sum_n = 0


class Edge:
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = 0


class Player:
    def __init__(self, action_size=81, channel=192):
        self.replay_memory = deque()
        self.action_size = action_size
        self.agent = PolicyValueNet(channel)
        self.tree = self.reset()

        self.change_temperature = 20
        self.epsilon = 0.25
        self.c_puct = 5
        self.dir_alpha = 0.2
        self.done = False
        self.win_index = None

    def reset(self):
        tree = defaultdict(Node)
        return tree

    def action(self, state):
        action = np.zeros(self.action_size)
        self.tree = self.reset()
        for i in range(1000):
            self.mcts(state, is_root_node=True)

        policy = self.get_policy(env)
        policy_temp = self.apply_temperature(policy, self.change_temperature)
        action_index = int(np.random.choice(action, p=policy_temp))
        action[action_index] = 1
        return policy, action

    def mcts(self, state, is_root_node=False):
        # return z
        if env.done:
            if env.win_index == 0:
                return 0
            return -1

        # expansion
        if state not in self.tree:
            leaf_p, leaf_v = self.expand_and_evaluate(state)
            # make edge!! need to be fixed
            self.tree[state].p = leaf_p
            # I'm returning everything from the POV of side to move
            return leaf_v

        else:
            # selection
            action = self.select_action_q_and_u(state, is_root_node)
            my_node = self.tree[state]
            my_edge = my_node.a[action]
            my_edge.q = my_edge.w / my_edge.n

            board, state, valid_pos, self.win_index, turn = env.step(action)
            leaf_v = self.mcts(state)  # next move from enemy POV
            leaf_v = -leaf_v

        # backup
        my_node.sum_n += 1
        my_edge.n += 1
        my_edge.w += leaf_v
        my_edge.q = my_edge.w / my_edge.n

        return leaf_v

    def expand_and_evaluate(self, state):
        leaf_p, leaf_v = self.agent.forward(state)
        return leaf_p, leaf_v

    def select_action_q_and_u(self, state, is_root_node):
        my_node = self.tree[state]

        best_s = -999
        best_a = None
        i = 0

        for action, a_s in my_node.a.items():
            policy = a_s.p
            if is_root_node:
                noise = np.random.dirichlet([self.dir_alpha] * len(my_node.a))
                policy = (1 - self.epsilon) * policy + self.epsilon * noise[i]
                i += 1
            ucb = a_s.q + self.c_puct * policy * np.sqrt(
                my_node.sum_n + 1) / (1 + a_s.n)
            if ucb > best_s:
                best_s = ucb
                best_a = action
        return best_a

    def apply_temperature(self, policy, turn):
        tau = np.power(self.play_config.tau_decay_rate, turn + 1)
        if tau < 0.1:
            tau = 0
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.labels_n)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1 / tau)
            ret /= np.sum(ret)
            return ret

    def get_policy(self, env):
        state = state_key(env)
        my_Node = self.tree[state]
        policy = np.zeros(self.action_size)
        for action, a_s in my_Node.a.items():
            policy[action] = a_s.n

        policy /= np.sum(policy)
        return policy

    def finish_game(self, z):
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]
