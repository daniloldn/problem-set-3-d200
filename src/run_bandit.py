from src.bandit_env import AdChannelBandit
from src.policy import run_random
from src.metrics import total_rewards, regret
from src.orchestrator import plot_avg_curves, run_experiment
import plotly.express as px
import numpy as np



def simulate(model):

    #bandit init
    bandit = AdChannelBandit()

    #time periods
    T=2000

    all_rewards = np.empty((50, T), dtype=float)
    all_regrets = np.empty((50, T), dtype=float)

    #run policy 
    for i in range(50):
        results_dict = model(bandit, T, seed= 42+i)
        rewards = total_rewards(results_dict["rewards"])
        regrets= regret(bandit, results_dict["arms"])

        #storing results
        all_rewards[i, :] = rewards
        all_regrets[i, :] = regrets

    avg_reward = all_rewards.mean(axis=0)
    avg_regret = all_regrets.mean(axis=0)

    fig1 = px.line(
    x=range(len(avg_reward)),
    y=avg_reward,
    title="Random policy: cumulative reward",
    labels={"x": "t", "y": "cumulative reward"}
    )
    fig1.show()

    fig2 = px.line(
    x=range(len(avg_regret)),
    y=avg_regret,
    title="Random policy: cumulative regret",
    labels={"x": "t", "y": "cumulative regret"}
    )
    fig2.show()

    return None

def simulate_2(models):

    rewards, regrets = run_experiment(models)

    #rewards

    plot_avg_curves(rewards, "Total rewards across models", "Cumaltive Rewards")
    plot_avg_curves(regrets, "Regret across models","Cumlative Regrets")


    return None

