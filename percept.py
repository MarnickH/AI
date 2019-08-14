class Percept:
    def __init__(self, state, action, new_state, reward):
        self.state = state
        self.action = action
        self.new_state = new_state
        self.reward = reward
