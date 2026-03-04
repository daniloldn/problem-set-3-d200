import numpy as np


class AdChannelBandit:
    """K-armed bandit for ad channel selection with Bernoulli rewards."""
    def __init__(self):
        self.channels = ["Email", "Social Media", "Display Ads", "Search", "Influencer"]
        self.K = len(self.channels)
        self._rates = np.array([0.08, 0.15, 0.12, 0.25, 0.05])
        self.seed = 42

    def pull(self, arm):
        """Select a channel and observe conversion (1) or not (0)."""
        np.random.default_rng(self.seed)
        return np.random.binomial(1, self._rates[arm])
    
    def optimal_arm(self):
        """Index of the channel with the highest true conversion rate."""
        return np.argmax(self._rates)
    
    def expected_reward(self, arm):
        """True conversion rate of the given channel (used for regret calculation)."""
        return self._rates[arm]
