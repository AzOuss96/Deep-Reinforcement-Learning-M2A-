from gym import wrappers, logger
import numpy as np
import copy
import json
import math


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()


class ValueIterationAgent:
    def __init__(self, mdp, statedic, state_id_dic, action_space):
        self.mdp = mdp
        self.action_space = action_space
        self.statedic = statedic
        self.state_id_dic = state_id_dic
        self.values = dict(zip([statedic[k] for k, _ in mdp.items()], [np.random.random() for i in range(len(mdp))]))
        self.policy = dict(zip([statedic[k] for k, _ in mdp.items()], [self.action_space.sample() for i in range(len(mdp))]))

    def learn(self, eps=0.000001, gamma = 0.7):
        while(1):
            values_copy = self.values.copy()
            for s in values_copy.keys():
                transitions = self.mdp[self.to_state(s)].items()
                rsum = {a : sum([p * (r + gamma * values_copy[self.to_state_id(s_prime)]) if done == False else r for (p, s_prime, r, done) in t]) for a, t in transitions}
                optimal_action = max(rsum.keys(), key=(lambda a: rsum[a]))
                self.policy[s] = optimal_action
                self.values[s] = rsum[optimal_action]
            if math.sqrt(sum([(values_copy[s] - self.values[s])**2 for s in values_copy.keys()])) < eps:
                break

    def reset(self):
            self.values = dict(zip([self.statedic[k] for k, _ in self.mdp.items()], [np.random.random() for i in range(len(self.mdp))]))
            self.policy = dict(zip([self.statedic[k] for k, _ in self.mdp.items()], [self.action_space.sample() for i in range(len(self.mdp))]))

    def act(self, state):
        state = self.statedic[json.dumps(state.tolist())]
        return self.policy[state]

    def to_state_id(self, state):
        return self.statedic[state]

    def to_state(self, state_id):
        return self.state_id_dic[state_id]


class PolicyIterationAgent:
    def __init__(self, mdp, statedic, state_id_dic, action_space):
        self.mdp = mdp
        self.action_space = action_space
        self.statedic = statedic
        self.state_id_dic = state_id_dic
        self.values = dict(zip([statedic[k] for k, _ in mdp.items()], [np.random.random() for i in range(len(mdp))]))
        self.policy = dict(zip([statedic[k] for k, _ in mdp.items()], [self.action_space.sample() for i in range(len(mdp))]))

    def learn(self, eps=0.000001, gamma = 0.7):
        while(1):
            policy_copy = self.policy.copy()

            while(1):
                values_copy = self.values.copy()

                for s in values_copy.keys():
                    transitions = self.mdp[self.to_state(s)]
                    action = self.policy[s]
                    t = transitions[action]
                    self.values[s] = sum([p * (r + gamma * values_copy[self.to_state_id(s_prime)]) if done == False else r for (p, s_prime, r, done) in t])

                if math.sqrt(sum([(values_copy[s] - self.values[s])**2 for s in values_copy.keys()])) < eps:
                    break

            for s in self.values.keys():
                transitions = self.mdp[self.to_state(s)].items()
                rsum = {a : sum([p * (r + gamma * self.values[self.to_state_id(s_prime)]) if done == False else r for (p, s_prime, r, done) in t]) for a, t in transitions}
                optimal_action = max(rsum.keys(), key=(lambda a: rsum[a]))
                self.policy[s] = optimal_action

            if self.policy == policy_copy :
                break

    def reset(self):
            self.values = dict(zip([self.statedic[k] for k, _ in self.mdp.items()], [np.random.random() for i in range(len(self.mdp))]))
            self.policy = dict(zip([self.statedic[k] for k, _ in self.mdp.items()], [self.action_space.sample() for i in range(len(self.mdp))]))

    def act(self, state):
        state = self.statedic[json.dumps(state.tolist())]
        return self.policy[state]

    def to_state_id(self, state):
        return self.statedic[state]

    def to_state(self, state_id):
        return self.state_id_dic[state_id]
