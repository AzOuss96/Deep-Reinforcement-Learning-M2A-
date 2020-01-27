from matplotlib import pyplot as plt
import seaborn as sns
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from modules.MAB import *


path = "./data/"
data = "CTR.txt"


if __name__ == "__main__":

    file = pd.read_csv(path+data, sep=':')
    contexts = np.vstack(list(map(lambda x: np.array(x.split(';')).astype(float), file.values[:,1])))
    rewards = np.vstack(list(map(lambda x: np.array(x.split(';')).astype(float), file.values[:,-1])))

    num_arms = rewards.shape[1]
    context_dim = contexts.shape[1]
    alpha = 1.0
    ucb = UCB(num_arms=num_arms, eps=0.6)
    linucb = LinUCB(alpha, num_arms, context_dim)

    random_choice = []
    optimal_choice = []
    ucb_choice = []
    linucb_choice = []
    iterations = np.arange(rewards.shape[0])

    for i in iterations:
        reward = rewards[i,:]
        context = np.stack(num_arms*[contexts[i,:]])

        ucb_choice.append(ucb.choose_arm(reward))
        optimal_choice.append(optimal(reward))
        random_choice.append(random(num_arms))
        linucb_choice.append(linucb.choose_arm(reward, context))



    ucb_rewards = np.cumsum(rewards[iterations,ucb_choice])
    random_rewards = np.cumsum(rewards[iterations,random_choice])
    optimal_rewards = np.cumsum(rewards[iterations,optimal_choice])
    linucb_rewards = np.cumsum(rewards[iterations,linucb_choice])

    ## Performance comparaison plots ##
    sns.lineplot(iterations, random_rewards)
    sns.lineplot(iterations, ucb_rewards)
    sns.lineplot(iterations, optimal_rewards)
    sns.lineplot(iterations, linucb_rewards)
    plt.legend(['Random', 'UCB', 'Optimal', r'$ LinUCB  \: (\alpha = %.f) $'%(alpha)])
    plt.xlabel('Iterations')
    plt.ylabel('Cumulated rewards')
    plt.show()


    ## Effect of alpha ##
    alphas = np.concatenate((np.arange(0,100,20)/100, np.arange(10, 100, 20)/10, np.arange(10,110,20)))
    alpha_rewards = []
    for alpha in alphas:
        r = 0
        linucb = LinUCB(alpha, num_arms, context_dim)
        for i in iterations:
            reward = rewards[i,:]
            context = np.stack(num_arms*[contexts[i,:]])
            j = linucb.choose_arm(reward, context)
            r += reward[j]
        alpha_rewards.append(r)

    max = np.max(alpha_rewards)
    ## Effect of alpha plots ##

    graph = sns.scatterplot(alphas,
                            alpha_rewards,
                            hue=(alpha_rewards == max).astype(float)
                            )

    graph.axhline(random_rewards[-1], ls=':')
    graph.axhline(ucb_rewards[-1], ls='-.')
    graph.axhline(optimal_rewards[-1], ls='--')
    plt.legend(['Random', 'UCB', 'Optimal', 'LinUCB'])
    plt.xlabel(r'$ \alpha \: values $')
    plt.ylabel('Final cumulated rewards')
    plt.show()
