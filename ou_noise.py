# --------------------------------------
# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# --------------------------------------

import numpy as np
import numpy.random as nr

class OUNoise:
    """docstring for OUNoise"""
    # def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.2):
    def __init__(self,action_dimension,mu=0, theta=0.2, sigma=0.8):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x)*0.01 + self.sigma * nr.randn(len(x))*np.sqrt(0.01)
        self.state = x + dx
        return np.clip(self.state,-1,1)

if __name__ == '__main__':
    ou = OUNoise(5)
    states = []
    for i in range(100):
        states.append(ou.noise())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
