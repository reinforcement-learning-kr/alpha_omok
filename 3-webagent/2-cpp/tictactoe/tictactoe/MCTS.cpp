//
//  MCTS.cpp
//  tictactoe
//
//  Created by 김태영 on 2018. 5. 18..
//  Copyright © 2018년 김태영. All rights reserved.
//

#include "MCTS.hpp"

MCTS::MCTS(int win_mark)
{
    _win_mark = win_mark;
}

MCTS::~MCTS()
{


}

int 
MCTS::Selection(Tree tree)
{
/*
node_id = (0,)

while True:
    num_child = len(tree[node_id]['child'])
    # Check if current node is leaf node
    if num_child == 0:
        return node_id
    else:
        max_value = -100
        leaf_id = node_id
        for i in range(num_child):
            action = tree[leaf_id]['child'][i]
            child_id = leaf_id + (action,)
            w = tree[child_id]['w']
            n = tree[child_id]['n']
            total_n = tree[tree[child_id]['parent']]['n']

            # for unvisited child, cannot compute u value
            # so make n to be very small number
            if n == 0:
                q = w / 0.0001
                u = 5 * np.sqrt(2 * np.log(total_n) / 0.0001)
            else:
                q = w / n
                u = 5 * np.sqrt(2 * np.log(total_n) / n)

            if q + u > max_value:
                max_value = q + u
                node_id = child_id

                */
    return 0;
}

int 
MCTS::Expansion(Tree tree, int leaf_id)
{
/*
leaf_state = deepcopy(tree[leaf_id]['state'])
is_terminal = check_win(leaf_state, self.win_mark)
actions = valid_actions(leaf_state)
expand_thres = 10

if leaf_id == (0,) or tree[leaf_id]['n'] > expand_thres:
    is_expand = True
else:
    is_expand = False

if is_terminal == 0 and is_expand:
    # expansion for every possible actions
    childs = []
    for action in actions:
        state = deepcopy(tree[leaf_id]['state'])
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
        tree[child_id] = {'state': state,
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

    */
    return 0;
}

int 
MCTS::Simulation(Tree tree, int child_id)
{
    /*

def simulation(self, tree, child_id):
state = deepcopy(tree[child_id]['state'])
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

            */
    return 0;
}

void 
MCTS::Backup(Tree tree, int child_id, int sim_result)
{
/*
def backup(self, tree, child_id, sim_result):
player = deepcopy(tree[(0,)]['player'])
node_id = child_id

# sim_result: 1 = O win, 2 = X win, 3 = Draw
if sim_result == 3:
    value = 0.8
elif sim_result - 1 == player:
    value = 1
else:
    value = -1

while True:
    tree[node_id]['n'] += 1
    tree[node_id]['w'] += value
    tree[node_id]['q'] = tree[node_id]['w'] / tree[node_id]['n']

    parent_id = tree[node_id]['parent']
    if parent_id == (0,):
        tree[parent_id]['n'] += 1
        return tree
    else:
        node_id = parent_id
        
        */
}