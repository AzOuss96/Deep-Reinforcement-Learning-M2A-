import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import gym
import gridworld
import seaborn as sns
import pandas as pd
import os,sys,inspect
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from modules.Iterations import *

# Instantiate the parser
parser = argparse.ArgumentParser()

parser.add_argument('agent_type', type=str,
                    help='in {value, policy}')

if __name__ == '__main__':

    args = parser.parse_args()
    if args.agent_type not in ['policy', 'iteration']:
        raise ValueError('agent argument should  be either policy or iteration')

    env = gym.make("gridworld-v0")
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    state_id_dic = {v:k for k,v in statedic.items()}

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    state_id_dic = {v:k for k,v in statedic.items()}
    env.seed()  # Initialiser le pseudo aleatoire

    if args.agent == 'iteration':
        agent = ValueIterationAgent(mdp, statedic, state_id_dic, env.action_space)
    elif args.agent == 'policy':
        agent = PolicyIterationAgent(mdp, statedic, state_id_dic, env.action_space)

    env.verbose = False
    gammas = [0.05, 0.99]
    episode_count = 7000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    data = []

    for gamma in gammas:
        agent.reset()
        agent.learn(gamma = gamma)

        for i in range(episode_count):
            obs = envm.reset()
            #env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100

            if env.verbose:
                env.render(FPS)

            j = 0
            rsum = 0
            
            while True:
                state = obs
                action = agent.act(obs)
                obs, reward, done, _ = envm.step(action)
                rsum += reward
                if (j % 3 == 0 and j > 0):
                    data.append(np.array([i, j, rsum, round(gamma, 2)]))
                j += 1

                if env.verbose :
                    env.render(FPS)(i % 100 == 0 and i > 0)

                if done:
                    print("Gamma : " + str(gamma) + ", Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                    break

    print("done")
    env.close()

    #################################
    ## Plotting performance curves ##
    #################################

    data = np.stack(data)
    data = pd.DataFrame(data, columns=['episode', 'iteration', 'rsum', 'gamma'])
    data["gamma"] = ["$%s$" % x for x in data["gamma"]]  ## Solves the issue regarding seaborn numerical hue

    ## Filtering trajectories based on the number of iterations
    data_ = data.rename({'iteration':'max iter'}, axis=1).drop(['rsum'], axis=1).groupby(['episode', 'gamma'], as_index=False)[['max iter']].apply(np.max).reset_index()
    data = data.merge(data_, on=['episode', 'gamma'])
    data = data.loc[data['max iter'] <= 105, ]
    data.drop(['max iter'], axis=1, inplace=True)
    data_ = None

    ## Smoothing the cumulated rewards
    data = data.sort_values(['episode', 'iteration', 'gamma'],ascending=False).groupby(['episode', 'iteration', 'gamma'])['rsum'].rolling(window=2, min_periods=1).mean().reset_index(['episode', 'iteration', 'gamma']).dropna()

    # plot
    sns.set()
    ax = sns.lineplot(x="iteration", y="rsum", hue="gamma",  data=data)
    plt.show()
