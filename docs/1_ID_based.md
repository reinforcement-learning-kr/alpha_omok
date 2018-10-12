# ID-based Implementation

Monte-Carlo Tree Search (MCTS) algorithm of Alpha Omok is implemented with ID-based method. This ID includes all the history of the Omok game with a single tuple, so implementation of MCTS with the ID has many advantages. 

<br>

## How to make ID

The ID is just sequence of the actions in the game. Let's assume the board is 3x3 size. The action index is like as follows. 

<p align= "center">
  <img src="./image/ID_index.png" width="300" alt="simple board example" />
</p>

The ID of empty board is **0** and the ID is made with a tuple. So, the ID of the first state in the game is **(0,)**, a tuple with a single element (0). The example process of making ID is as follows.

<p align= "center">
  <img src="./image/ID_process.png" width="800" alt="simple board example" />
</p>

Alpha Omok has 2 different kinds of environments. First one is **env_small**, which size is 9x9. In this case, action index will be **0 ~ 80**, because the number of all the available actions is 81. The second environment is **env_regular**, which size is 15x15. In this case, action index will be **0 ~ 224**, because the number of all the available actions is 225. 

<br>

## Advantages of using ID

The advantages of using ID is as follows. 

- It doesn't need to save all the state of the game board. 
- All the nodes of MCTS can be distinguished. 
- All the legal actions can be obtained. 



At first, AlphaGo Zero algorithm needs state of the game as an input of network to obtain policy and value. The state is not just current state of the board. According to the AlphaGo Zero paper, the state contains the information as follows. 

<p align= "center">
  <img src="./image/state.png" width="500" alt="state of AlphaGo Zero" />
</p>

Therefore, to make each state, we need to know history of the game state. Also, all the states need to be saved in the dataset to train the algorithm. If we save state of many games, large size of memory is needed. However, the problems are easily solved with the ID. We save the ID instead of state. If the environment is env_regular, the size of state is (15x15x17), but the size of ID is 226 at maximum. 

<br>

Also, all the nodes of MCTS can be distinguished with the ID, becuase all the ID of each nodes are different. We don't need to save the state of board for each node during the MCTS process. The states can be rebuilt using the ID. This also saves the memory when operating MCTS process. 

<br>

Lastly, valid actions can be obtained using the ID. 

```python
def legal_actions(node_id, board_size):
    all_action = {a for a in range(board_size**2)}
    action = set(node_id[1:])
    actions = all_action - action

    return list(actions)
```

The above code is contained in `util.py`. Using the code, we can obtain the valid action. 