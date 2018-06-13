//
//  MCTS.hpp
//  tictactoe
//
//  Created by 김태영 on 2018. 5. 18..
//  Copyright © 2018년 김태영. All rights reserved.
//

#ifndef MCTS_hpp
#define MCTS_hpp

#include <stdio.h>
#include "Tree.hpp"

class MCTS 
{
public:
    MCTS(int win_mark);
    ~MCTS();

public:
    int Selection(Tree tree);
    int Expansion(Tree tree, int leaf_id);
    int Simulation(Tree tree, int child_id);
    void Backup(Tree tree, int child_id, int sim_result);

private:

    int _win_mark;
};

#endif /* MCTS_hpp */
