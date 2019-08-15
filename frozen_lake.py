def set_chances(env):
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            transitions = env.env.P[state][action]
            if len(transitions) == 3:
                for transition in range(3):
                    p, s, r, d = transitions[transition]
                    if transition != 1:
                        transitions[transition] = (0.1, s, r, d)
                    else:
                        transitions[transition] = (0.8, s, r, d)
