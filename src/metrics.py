import numpy as np


def total_rewards(rewards):
    return np.cumsum(rewards)

def regret(bandit, arms):

    #optimal choice
    p_star = bandit.expected_reward(bandit.optimal_arm())

    #probability of chosen arm
    p_chosen = np.array([bandit.expected_reward(a) for a in arms ])

    return np.cumsum(p_star - p_chosen)