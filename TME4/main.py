import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
%matplotlib inline
import gym
import gridworld
from gym import wrappers, logger
import torch.optim as optim
import pandas as pd
from torch import nn
import seaborn as sns
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from modules.utils import ReplayMemory
from modules.DQN import DQNAgent


if __name__ == '__main__':

    ### parameters ###
    BUFFER_SIZE = 200
    BATCH_SIZE = 20

    GAMMA = 0.7

    EPS = 0.2
    EPS_DECAY = 0.00001
    UPDATE_FREQ = 20


    inSize  = 8
    outSize = 4
    layers = [24, 24]

    ### Environment setting ###
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    outdir = env_name+'/DQN-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)
    env.reset()
    done = False
    verbose = False


    ### Initialization ###  DQN + Experience replay
    replay_mem = ReplayMemory(BUFFER_SIZE, BATCH_SIZE)
    dqn_target = DQNAgent(inSize, outSize, layers, eps=EPS, eps_decay=EPS_DECAY).double()
    dqn_agent = DQNAgent(inSize, outSize, layers).double()


    ### Training Settings ###
    episode_count = 1000
    optimizer = optim.Adam(dqn_agent.parameters())
    huberLoss = nn.SmoothL1Loss()
    rsum_hist = []

    ### Training loop ###

    for episode in range(episode_count):

        it = 0
        obs = envm.reset()

        rsum = 0.0
        done = False

        while(True):

            if verbose == True:
                env.render()

            state = obs
            action = dqn_agent.act(state, env)
            obs, reward, done, _ = envm.step(action)
            rsum += reward


            ### storing experience ###
            replay_mem.store(state, action, reward, obs, done)

            if len(replay_mem) > BATCH_SIZE:

                ### sampling batchs ###
                states, actions, rewards, next_states, dones = replay_mem.sample()

                ### Updating DQN agent ###
                q_targets = rewards + GAMMA * dqn_target.evaluate_max(next_states) * (1.0 - dones.double())

                ### forward + Computing loss + backprop ###
                optimizer.zero_grad()
                q_estimates = dqn_agent.evaluate(states, actions)
                loss = huberLoss(q_estimates, q_targets)
                loss.backward()
                optimizer.step()

            ### Resetting target DQN ###
            if it % UPDATE_FREQ == 0:
                dqn_target = dqn_agent

            if done == True:
                print("Episode : " + str(episode) + " rsum=" + str(rsum) +  " iter = "+ str(it) + "eps = " + str(EPS))
                rsum_hist.append(rsum)
                break

            it += 1

    env.close()


    ## Plotting cumulative reward curves ##
    window = 70
    rsum_hist = pd.concat([pd.Series(rsum_hist, name='mean').rolling(window).mean(),
           pd.Series(rsum_hist, name='std').rolling(window).std()],
          axis=1)

    sns.set()
    plt.figure(figsize=(10,5))
    ax = rsum_hist['mean'].plot()

    ax.fill_between(rsum_hist.index, rsum_hist['mean'] - rsum_hist['std'], rsum_hist['mean'] + rsum_hist['std'],
                    alpha=.25)
    plt.tight_layout()
    plt.ylabel("Cumulated sum of rewards (Y)")
    plt.title('%s :  $\gamma = %.2f$ | $\epsilon = %.2f$ | $ update \: rate \:(C) = %d$'%(env_name, GAMMA, EPS, UPDATE_FREQ))
    plt.legend(['70-Episod rolling mean of Y', '70-Episod rolling std of Y'])
    plt.xlabel('Episod')
    sns.despine()

    plt.savefig('./plots/'+env_name)
