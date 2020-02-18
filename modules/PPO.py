import torch
from torch import nn
from torch import *
import torch.nn.functional as F

class PPO:
    def __init__(self, model, KL_penalty, KL_target, eps, clip=False):
        self.KL_penalty = KL_penalty
        self.KL_target = KL_target
        self.model = model
        self.eps = eps
        self.clip = clip
        self._distprobs = None

    def compute_loss(self, states, actions, rewards, obs, dones, gamma):

        advantages = [self.model.compute_advantage(s, a, r, o, d, gamma) for (s, a, r, o, d) in zip(states, actions, rewards, obs, dones)]
        dist_probs = [self.model.compute_distprobs(s, a) for (s, a) in zip(states, actions)]

        ratios = [dist / prev_dist for dist, prev_dist in zip(dist_probs, self._distprobs)]

        advantage_loss = torch.stack([- (r * a).mean() for r, a in zip(ratios, advantages)]).mean()

        KL_loss = torch.stack([F.kl_div(dist.log(), prev_dist) for dist, prev_dist in zip(dist_probs, self._distprobs)]).mean()

        actor_loss = torch.stack([self.model.compute_loss(s, a, r, o, d, gamma)[1] for (s, a, r, o, d) in zip(states, actions, rewards, obs, dones)]).mean()

        if self.clip == False:

            critic_loss = advantage_loss + self.KL_penalty *  KL_loss

            if KL_loss >= 1.5 * self.KL_target:
                self.KL_penalty = 2 * self.KL_penalty

            if KL_loss <=  self.KL_target / 1.5:
                self.KL_penalty = self.KL_penalty / 2

        elif self.clip == True:

            eps = self.eps
            critic_loss = torch.stack([- torch.sum(torch.min(torch.stack([r*a, torch.clamp(r, 1-eps, 1+eps)*a], axis=-1), axis=-1).values) for r, a in zip(ratios, advantages)]).mean()

        return critic_loss, actor_loss


    def save_distprobs(self, states, actions):
        with torch.no_grad():
            self._distprobs = [self.model.compute_distprobs(s, a) for (s, a) in zip(states, actions)]
