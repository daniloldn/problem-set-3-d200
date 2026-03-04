from src.bandit_env import AdChannelBandit
from src.policy import run_random
from src.metrics import total_rewards, regret
import plotly.express as px



def simulate():

    #bandit init
    bandit = AdChannelBandit()

    #time periods
    T=2000

    #run policy 
    results_dict = run_random(bandit, T)

    rewards = total_rewards(results_dict["rewards"])
    regrets= regret(bandit, results_dict["arms"])

    fig1 = px.line(
    x=range(len(rewards)),
    y=rewards,
    title="Random policy: cumulative reward",
    labels={"x": "t", "y": "cumulative reward"}
    )
    fig1.show()

    fig2 = px.line(
    x=range(len(regrets)),
    y=regrets,
    title="Random policy: cumulative regret",
    labels={"x": "t", "y": "cumulative regret"}
    )
    fig2.show()

    return None

