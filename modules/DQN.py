import numpy as np
from torch import nn
from torch import FloatTensor
from torch import *
import torch
import random


class DQNAgent(nn.Module):
    def __init__(self, inSize, outSize, layers, eps=0.2, eps_min=0.1,eps_decay=0.9999):
        super(DQNAgent, self).__init__()
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))


    def forward(self, x):
        x = x.double()
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x.double())
            x = self.layers[i](x).double()
        return x

    def act(self, state, env):
        if np.random.rand() < self.eps:
            action = env.action_space.sample()
        else :
            action = self.forward(torch.tensor(state)).detach().max(-1).indices.item()
        if self.eps > self.eps_min :
            self.eps = self.eps *  self.eps_decay
        return action

    def evaluate(self, state, action):
        '''returns Qvalue of state, action'''
        return self.forward(state).gather(1, action.reshape(-1, 1)).view(-1)

    def evaluate_max(self, state):
        '''returns the max Qvalue of state '''
        return self.forward(state).detach().max(-1).values
