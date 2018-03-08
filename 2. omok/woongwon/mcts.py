class MCTS:
    def selection(self, tree, model):
        return leaf_state

    def expansion(self, tree, model, leaf_state):
        action_probs, value = model.forward(leaf_state)
        # initialize new edges with action probs
        return tree, value

    def backup(self, tree, model, leaf_id, value):
        return tree
