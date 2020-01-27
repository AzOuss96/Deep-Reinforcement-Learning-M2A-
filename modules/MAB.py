import numpy as np
import pandas as pd


def random(num_arms):
    return np.random.randint(1, num_arms)

def staticbest(arr):
    return np.argmax(np.cumsum(arr, axis=1), axis=0)

def optimal(arr):
    return np.argmax(arr, axis=0)

class UCB:
    def __init__(self, num_arms, eps=0.6):
        self.t = 1
        self.eps = eps
        self.num_arms = num_arms
        self.rewards_hist = np.empty((0,num_arms))
        self.s = np.ones(num_arms)

    def choose_arm(self, rewards):
        self.rewards_hist = np.vstack((self.rewards_hist,
                  rewards))
        if np.random.random() > self.eps :
            i = random(self.num_arms)
        else :
            upper_bounds = np.mean(self.rewards_hist, axis=0) + np.sqrt(2*np.log(self.t)/self.s)
            i = optimal(upper_bounds)

        self.s[i] += 1
        self.t += 1

        return i

class LinUCB:
    def __init__(self, alpha, num_arms, context_dim):

        self.alpha = alpha
        self.num_arms = num_arms
        self.A = np.stack(num_arms * [np.identity(context_dim)])
        self.b = np.zeros((num_arms, context_dim, 1))
        self.theta = np.linalg.inv(self.A) @ self.b

    def choose_arm(self, rewards, context):
        p =  [self.theta.squeeze(2).dot(context[i]) + self.alpha * np.sqrt((context[i].dot(np.linalg.inv(self.A))@context[i])) for i in range(self.num_arms)]
        i = np.argmax(p)
        self.A[i] += context[i].T.dot(context[i])
        self.b[i] += rewards[i] * context[i].reshape(-1, 1)
        self.theta = np.linalg.inv(self.A) @ self.b
        return i
