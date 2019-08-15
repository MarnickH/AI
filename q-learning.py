import numpy as np
import gym
import random
import frozen_lake

env = gym.make("FrozenLake-v0")
frozen_lake.set_chances(env)

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

rewards = []

for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    while not done:
        explore = random.uniform(0, 1)
        if explore > epsilon:
            action = np.argmax(qtable[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        qtable[state, action] = qtable[state, action] + learning_rate * (
                reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

        total_rewards += reward

        state = new_state

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

print("Score over time: " + str(sum(rewards) / total_episodes))
print(qtable)

with open('qtable.txt', 'wb') as f:
    np.savetxt(f, qtable, fmt='%1.10f')

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
