#!/usr/bin/env python
# coding: utf-8

# In[161]:


import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn
import gym
#import gridworld
from gym import wrappers, logger
import numpy as np
import copy
from torch import nn
from torch import FloatTensor
from torch import *
import torch
import random
from collections import namedtuple, deque
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.distributions as tdist
from collections import OrderedDict


# In[37]:


BUFFER_SIZE = 100
BATCH_SIZE = 10

class ExperienceReplay(object):
    
    def __init__(self, buffer_size = BUFFER_SIZE, batch_size = BATCH_SIZE):
        super().__init__()
        self.batch_size = batch_size
        self.buffer = deque(maxlen = buffer_size)
        self.experience =  namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def store(self, experience):
        self.buffer.append(experience)
        
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        state = torch.stack([torch.tensor(exp.state) for exp in batch])
        action = torch.stack([torch.tensor(exp.action) for exp in batch])
        reward = torch.stack([torch.tensor(exp.reward) for exp in batch])
        next_state = torch.stack([torch.tensor(exp.next_state) for exp in batch])
        done = torch.stack([torch.tensor(exp.done) for exp in batch])
        
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
    
    


# In[38]:


class Net(nn.Module):
    
    def __init__(self, inSize, outSize, layers):
        super(Net, self).__init__()
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



# In[192]:


if __name__ == '__main__':

    ### parameters ###
    BUFFER_SIZE = 1000
    BATCH_SIZE = 100

    GAMMA = 0.9
    
    RHO = 0.4 #Polyak averaging

    ### Environment setting ###
    env = gym.make('MountainCarContinuous-v0')
    outdir = 'MountainCarContinuous-v0/ddpg-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)
    env.reset()
    done = False
    
    outSize = 1
    layers_actors = [4, 8, 4]
    layers_critics = [4, 8, 4]
    inSize_actor = envm.reset().size
    inSize_critic = inSize_actor + 1
    
    ### Initialization ###  Policy and value networks + Experience replay
    exp_replay = ExperienceReplay(BUFFER_SIZE, BATCH_SIZE)
    
    target_critic = Net(inSize_critic, outSize, layers_critics).double()
    critic = Net(inSize_critic, outSize, layers_critics).double()
    
    target_actor =  Net(inSize_actor, outSize, layers_actors).double()
    actor =  Net(inSize_actor, outSize, layers_actors).double()

    
    ### Training Settings ###
    episode_count = 100000
    
    #params = list(list(critic.parameters()) + list(actor.parameters()))
    
    optimizer_actor = optim.Adam(actor.parameters())
    optimizer_critic = optim.Adam(critic.parameters())
    
    MSE_loss = nn.MSELoss()
    rsum_hist = []
    
    # exploration noise
    noise = tdist.Normal(torch.tensor([0.0]), torch.tensor([0.3]))

    ### Training loop ###

    for episode in range(episode_count):

        it = 0
        obs = envm.reset()

        rsum = 0.0
        done = False

        while(True):

            state = obs
               
            action = actor.forward(torch.tensor(state)).detach() + noise.sample([1,]).squeeze(0)
            
            action =  torch.clamp(action, env.action_space.low[0], env.action_space.high[0]).detach()
            obs, reward, done, _ = envm.step(action)

            rsum += reward

            env.render()

            ### storing experience ###
            exp = exp_replay.experience(state, action, reward, obs, done)
            exp_replay.store(exp)

            if len(exp_replay) > BATCH_SIZE:

                ### sampling batchs ###
                states, actions, rewards, next_states, dones = exp_replay.sample()
            
                ### Updating DDPG agent ###
                optimal_actions = target_actor.forward(next_states).detach()
                
                states_optimal_actions = torch.cat((next_states, optimal_actions), dim=1).detach()
                
                q_targets = rewards.reshape(BATCH_SIZE, -1) + GAMMA * target_critic.forward(states_optimal_actions).detach() * (1.0 - dones.double().reshape(BATCH_SIZE, -1))
                

                ### actor : forward + Computing loss + backprop ###
                optimizer_critic.zero_grad()
                
                states_actions = torch.cat((states, actions), dim=1)

                q_estimates = critic.forward(states_actions)

                critic_loss = MSE_loss(q_estimates, q_targets.detach())
                
                critic_loss.backward()

                optimizer_critic.step()
                
                ### critic : forward + Computing loss + backprop ###
                optimizer_actor.zero_grad()
                
                actor_actions = actor.forward(states)
                
                states_actor_actions = torch.cat((states, actor_actions), dim=1)

                q_values = critic.forward(states_actor_actions)

                actor_loss = - torch.mean(q_values)
                
                actor_loss.backward()

                optimizer_actor.step()
                
                
                


            critic_new_state_dict = OrderedDict({name: param.detach() * RHO + (1.0 - RHO) * critic.state_dict()[name].detach() for name, param in target_critic.state_dict().items()})    
            target_critic.load_state_dict(critic_new_state_dict, strict=False)
                
            actor_new_state_dict = OrderedDict({name: param.detach() * RHO + (1.0 - RHO) * actor.state_dict()[name].detach() for name, param in target_actor.state_dict().items()})    
            target_actor.load_state_dict(actor_new_state_dict, strict=False)
 

            if done == True:
                print("Episode : " + str(episode) + " rsum=" + str(rsum) +  " iter = "+ str(it))
                rsum_hist.append(rsum)
                break


            it += 1


    env.close()

