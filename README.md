# Alpha-Omok
This is a project of [Reinforcement Learning KR group](https://www.facebook.com/groups/ReinforcementLearningKR/).



## Project objective
AlphaZero is Reinforcement Learning algorithm which is effectively combine [MCTS(Monte-Carlo Tree Search)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) with Actor-Critic. Our objective is to train [AlphaZero](https://deepmind.com/blog/alphago-zero-learning-scratch/) and apply it to [Omok (Gomoku)](https://en.wikipedia.org/wiki/Gomoku) 

There are 4 little objectives to achieve  
1. MCTS on Tic-Tac-Toe
2. MCTS on Omok
3. AlphaZero on Omok
4. Upload AlphaGo Zero on web



## Description of the Folders

### 1_tictactoe_MCTS

![tictactoe](./image/tictactoe.PNG)

 This folder is for implementing MCTS in [Tic-Tac-Toe](https://en.wikipedia.org/wiki/Tic-tac-toe). If you want to study MCTS only, please check the files in this folder. <br>

The description of the files in the folder is as follows. (files with bold text are codes for implementation)

- env: Tic-Tac-Toe environment code (made with [pygame](https://www.pygame.org/news))
- **mcts_guide**: MCTS doesn't play the game, it only recommends how to play. 
- **mcts_vs**: User can play with MCTS algorithm. 
- utils: functions for implementing algorithm. 



### 2_AlphaOmok

![omok](./image/omok.PNG)

  The folder is for implementing AlphaZero algorithm in omok environment. There are two versions of omok (9x9, 15x15). <br>

 The description of the files in the folder is as follows. (files with bold text are codes for implementation)

- **eval_local**: code for evaluating the algorithm on local PC
- **eval_server**: code for evaluating the algorithm on web
- **main**: main training code of Alpha Zero. 
- model: Network model (PyTorch)
- utils: functions for implementing algorithm. 



### Visualization

http://127.0.0.1:5000/gameboard_view
http://127.0.0.1:5000/agent_view/player/visit
http://127.0.0.1:5000/agent_view/player/pi
http://127.0.0.1:5000/agent_view/enemy/visit
http://127.0.0.1:5000/agent_view/enemy/pi



## Reference

1. [Mastering the Game of Go with Deep Neural Networks and Tree Search](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
2. [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270)

