import gym
import numpy as np 
from custom import CustomAutoencoder

steps = 10000000
test_steps = 5000
reduce_dim = 10

env = gym.make("Humanoid-v2")
data = []
test_data = []

# collect training data
print("collecting training data")
obs = env.reset()
print(len(obs))
for i in range(steps):
    data.append(obs[:269])
    action = env.action_space.sample()
    obs,r,d,_ = env.step(action)
    
    if i % 1000 == 0:
        print(i)
    
# collect test data
print("collecting testing data")
obs = env.reset()
for i in range(test_steps):
    test_data.append(obs[:269])
    action = env.action_space.sample()
    obs,r,d,_ = env.step(action)
    
    if i % 1000 == 0:
        print(i)
    
autoencoder = CustomAutoencoder(len(data[0]), 
                                num_hid1=128,
                                num_hid2=64,
                                reduce_dim=32, 
                                normalize=False)
autoencoder.set_data(data)
autoencoder.set_test_data(test_data)
autoencoder.train(testiter=20)
