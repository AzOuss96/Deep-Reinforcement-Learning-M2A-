import numpy as np
import torch
import random
from collections import namedtuple, deque
import torch
from torch import nn


BUFFER_SIZE = 100
BATCH_SIZE = 10

class ReplayMemory(object):

    def __init__(self, buffer_size = None, batch_size = BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
        self.buffer = deque(maxlen = buffer_size)
        self.experience =  namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def store(self, states, actions, rewards, obs, dones):
        experience = self.experience(states, actions, rewards, obs, dones)
        self.buffer.append(experience)

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        state = torch.stack([torch.tensor(exp.state) for exp in batch])
        action = torch.stack([torch.tensor(exp.action) for exp in batch])
        reward = torch.stack([torch.tensor(exp.reward).double() for exp in batch])
        next_state = torch.stack([torch.tensor(exp.next_state) for exp in batch])
        done = torch.stack([torch.tensor(exp.done) for exp in batch])

        return state, action, reward, next_state, done

    def pop(self, batch_size, stack=True):

        batch = [self.buffer.pop() for i in range(batch_size)]

        if stack :
            state = torch.stack([torch.tensor(exp.state) for exp in batch])
            action = torch.stack([torch.tensor(exp.action) for exp in batch])
            reward = torch.stack([torch.tensor(exp.reward).double() for exp in batch])
            next_state = torch.stack([torch.tensor(exp.next_state) for exp in batch])
            done = torch.stack([torch.tensor(exp.done) for exp in batch])
        else :
            state = [torch.tensor(exp.state) for exp in batch]
            action = [torch.tensor(exp.action) for exp in batch]
            reward = [torch.tensor(exp.reward).double() for exp in batch]
            next_state = [torch.tensor(exp.next_state) for exp in batch]
            done = [torch.tensor(exp.done) for exp in batch]

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer.clear()


class NeuralNet(nn.Module):

    def __init__(self, inSize, outSize, layers, soft = True):
        super(NeuralNet, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))
        if soft:
            self.layers.append(nn.Softmax(dim=-1))

    def forward(self, x):
        x = x.double()
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x.double())
            x = self.layers[i](x).double()
        return x
