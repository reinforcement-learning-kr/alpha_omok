# How to use eval local

## Settings

At the `eval_local.py`, you should make a few settings. 

<p align= "center">
  <img src="./image/eval_local.png" width="700" alt="simple board example" />
</p>

There are **player** and **enemy** for the evaluation. 

Player and enemy can be decided as follows. 

- human: human can play the game
- uct: play Omok with pure MCTS based on UCB 
- puct: play Omok with pure MCTS based on PUCT
- random: play Omok without any algorithm, it just play with random moves
- model path: If you write path of the model, it plays Omok with AlphaOmok algorithm with the variables that you set. 



If you want to test saved AlphaOmok model, you should set **N_BLOCKS**, **IN_PLANES** and **OUT_PLANES** according to the model. You can refer to the `log file` for finding those paramters of the model.



**N_MCTS** is the number of MCTS iterations. 4 steps of MCTS is 1 iteration. 4 steps of MCTS are as follows. 

\- Selection 

\- Expansion

\- Simulation (AlphaGo zero doesn't have this process)

\- Back up 

Therefore, 400 N_MCTS represents 400 iterations of MCTS. 



**N_MATCH** is the number of the games for testing. 



 ## Testing

If you finished all the settings, just run the `eval_local`. Then you can see the Omok playing between the agents or you can play Omok against the agent. You can see the demo that I played Omok with the AlphaOmok model as follows. 

 <p align= "center">
  <img src="./image/AgentWin1_speed.gif" width="300" alt="simple board example" />
</p>













