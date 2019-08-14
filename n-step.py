import numpy as np
import gym
import random
from percept import Percept

env = gym.make("FrozenLake-v0")

action_size = env.action_space.n
state_size = env.observation_space.n

filename = "qtable.txt"

qtable = np.zeros((state_size, action_size))
print(qtable)

total_episodes = 100000
learning_rate = 0.8
gamma = 0.95

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.0005
n = 5

rewards = []
p = []

for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    while not done:
        step += 1

        explore = random.uniform(0, 1)
        if explore > epsilon:
            action = np.argmax(qtable[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        new_percept = Percept(state, action, new_state, reward)
        p.append(new_percept)

        if len(p) >= n:
            for percept in p[step - n:step]:
                qtable[percept.state, percept.action] = qtable[percept.state, percept.action] - learning_rate * (
                            qtable[percept.state, percept.action] - (
                                percept.reward + gamma * np.max(qtable[percept.new_state, :])))

            total_rewards += reward

            state = new_state

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        rewards.append(total_rewards)

    print("Score over time: " + str(sum(rewards) / total_episodes))
    print(qtable)

    env.reset()

    for episode in range(5):
        state = env.reset()
        step = 0
        done = False
        print("****************************************************")
        print("EPISODE ", episode)

        while not done:
            step += 1
            action = np.argmax(qtable[state, :])

            new_state, reward, done, info = env.step(action)

            if done:
                env.render()
                print("Number of steps", step)
                break
            state = new_state
    env.close()
