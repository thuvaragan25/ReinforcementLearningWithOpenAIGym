import gym
import os
import time 
import numpy as np
import random

env = gym.make("Taxi-v3")

with open('qtable100000.npy', 'rb') as f:
    Q_table = np.load(f)

for i in range(5): 
    observation = env.reset()
    done = False
    while not done:
        print(env.render(mode='ansi'))  
        print("Episodes: " + str(i))
        action = np.argmax(Q_table[observation])     
        observation, reward, done, info = env.step(action)
        time.sleep(0.5)
        os.system("cls")
env.close()
