from src.bandit_env import AdChannelBandit
from src.policy import run_random
from src.metrics import total_rewards, regret
import plotly.express as px



def main():

    #bandit init
    bandit = AdChannelBandit()

    #time periods
    T=2000

    #run policy 
    results_dict = run_random(bandit, T)

    total_rewards = total_rewards(results_dict["rewards"])
    regret = regret(bandit, results_dict["arms"])

    #plots
    fig1 = px.line(total_rewards, title="Random policy: cumulative reward")
    fig1.show()

    fig2 = px.line(regret, title="Random policy: cumulative regret")
    fig2.show()

    return None

if __name__ == "__main__":
    main()