import gym
import numpy as np 
import os

if not os.path.exists("data"):
    os.mkdir("data")

steps = 20000000
test_steps = 5000

env = gym.make("Ant-v2")
ant_data = []
ant_test_data = []

# collect training data
print("collecting Ant training data")
obs = env.reset()
print(len(obs))
for i in range(steps):
    ant_data.append(obs[:29])
    action = env.action_space.sample()
    obs,r,d,_ = env.step(action)
    
    if i % 1000 == 0:
        print(i)
    
# collect test data
print("collecting Ant testing data")
obs = env.reset()
for i in range(test_steps):
    ant_test_data.append(obs[:29])
    action = env.action_space.sample()
    obs,r,d,_ = env.step(action)
    
    if i % 1000 == 0:
        print(i)

np.save("data/ant_data", ant_data)
np.save("data/ant_test_data", ant_test_data)

env = gym.make("Humanoid-v2")
humanoid_data = []
humanoid_test_data = []

# collect training data
print("collecting Humanoid training data")
obs = env.reset()
print(len(obs))
for i in range(steps):
    humanoid_data.append(obs[:269])
    action = env.action_space.sample()
    obs,r,d,_ = env.step(action)
    
    if i % 1000 == 0:
        print(i)
    
# collect test data
print("collecting Humanoid testing data")
obs = env.reset()
for i in range(test_steps):
    humanoid_test_data.append(obs[:269])
    action = env.action_space.sample()
    obs,r,d,_ = env.step(action)
    
    if i % 1000 == 0:
        print(i)

np.save("data/humanoid_data", humanoid_data)
np.save("data/humanoid_test_data", humanoid_test_data)