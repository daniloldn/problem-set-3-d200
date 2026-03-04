import numpy as np 
import random as rng



def run_random(bandit, T=200, seed=42):
    """Run a random policy for T rounds.

    Returns:
    rewards: array of length T, reward at each round (0 or 1)
    arms_played: array of length T, index of arm chosen each round
"""
    #setting seed fo reproducabilty 
    np.random.seed(seed)
    #storing policy and rewards at each period
    arms = np.empty(T, dtype=int)
    rewards = np.empty(T, dtype=int)

    #interating through each period
    for t in range(T):
        #selects a arm at random then sees what reward it gets
        a = np.random.randint(0, bandit.K)
        r = bandit.pull(a)

        #stores results
        arms[t] = a
        rewards[t] = r

    return {"arms": arms, "rewards": rewards}


def e_greedy(bandit, T=200, seed=42, epsilon=0.1,):

    #setting seed fo reproducabilty 
    np.random.seed(seed)
    #storing policy and rewards at each period
    arms = np.empty(T, dtype=int)
    rewards = np.empty(T, dtype=int)

    #keeping track of what has happened
    N = np.zeros(bandit.K, dtype=int)
    Q = np.zeros(bandit.K, dtype=float)

    for t in range(T):
        u = np.random.uniform(0,1)
        if u < epsilon:
            a = np.random.randint(0, bandit.K)
        else:
            a = int(np.argmax(Q))

        r = bandit.pull(a)
        N[a] += 1
        Q[a] = (Q[a]* (N[a] -1) + r)/N[a]

       #stores results
        arms[t] = a
        rewards[t] = r


    return {"arms": arms, "rewards": rewards}