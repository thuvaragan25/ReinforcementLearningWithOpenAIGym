import gym
import os
import numpy as np
import random

env = gym.make("Taxi-v3")

Q_table = np.zeros((env.observation_space.n, env.action_space.n))

lr = 0.1
discount = 0.6
epsilon = 0.1

for i in range(100000):
    observation = env.reset()
    done = False
    while not done:
        print(env.render(mode='ansi')) 
        print("Episodes: " + str(i))

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() 
        else:
            action = np.argmax(Q_table[observation])
        
        new_observation, reward, done, info = env.step(action)
        Q_table[observation, action] = (1-lr) * Q_table[observation, action] + lr * (reward + discount * np.max(Q_table[new_observation]))

        observation = new_observation        
        if os.name == 'nt': 
            os.system('cls')
        else: 
            os.system('clear')

env.close()
with open('qtable100000.npy', 'wb') as f:
    np.save(f, Q_table)




