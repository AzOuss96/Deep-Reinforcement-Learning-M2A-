import numpy as np
from torch import nn
from torch import FloatTensor
from torch import *
import torch
from torch.distributions import Categorical



class A2CAgent(nn.Module):
    def __init__(self, actor, critic):
        super(A2CAgent, self).__init__()
        self.actor = actor
        self.critic = critic

    def act(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action

    def evaluate(self, state, action):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(state)

        return action_logprobs, state_value.squeeze(-1), dist_entropy

    def compute_loss(self, states, actions, rewards, obs, dones, gamma):
        huberLoss = nn.SmoothL1Loss()

        next_actions = self.act(obs)
        _, next_values, _ = self.evaluate(obs, next_actions)
        logprobs, values, dist_entropy  = self.evaluate(states, actions)

        ## evaluating advantages ##
        returns = rewards + gamma * next_values * (1.0 - torch.tensor(dones).double())
        advantages = returns - values

        ## critic loss ##
        critic_loss = huberLoss(returns, values)

        ## actor loss ##
        actor_loss = - (advantages.detach() * logprobs).mean()

        return critic_loss, actor_loss

    def compute_advantage(self, states, actions, rewards, obs, dones, gamma):
        next_actions = self.act(obs)
        _, next_values, _ = self.evaluate(obs, next_actions)
        _, values, _  = self.evaluate(states, actions)
        returns = rewards + gamma * next_values * (1.0 - torch.tensor(dones).double())
        advantages = returns - values
        return advantages

    def compute_distprobs(self, states, actions):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states)

        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)

        action_probs = self.actor(states)
        return torch.gather(action_probs, 1, actions.view(-1,1))
