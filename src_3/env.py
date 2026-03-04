import gymnasium
import numpy as np
from gymnasium import spaces
from itertools import combinations
from scipy.special import expit
class ContentPersonalizationEnv(gymnasium.Env):
    """
Within-session content personalisation.
A platform displays K=3 of M=5 content categories at each page view.
The visitor (one of 3 latent types) clicks according to an MNL model
over the displayed categories plus an outside option. At session end,
conversion probability depends on how well displayed content matched
the visitor's true preferences.
"""
    CATEGORIES = ["Tech", "Fashion", "Sports", "Food", "Travel"]
    TYPE_NAMES = ["Tech enthusiast", "Fashionista", "Sports fan"]
    TYPE_UTILITIES = np.array([
    [2.0, 0.2, 0.5, 0.3, 0.8], # Type 0
    [0.3, 2.0, 0.2, 0.8, 1.0], # Type 1
    [0.5, 0.3, 2.0, 0.5, 0.2], # Type 2
    ])
    OUTSIDE_UTILITY = 0.5

    def __init__(self, max_steps=8):
        super().__init__()
        self.max_steps = max_steps
        self.n_categories = 5
        self.assortments = list(combinations(range(5), 3))
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.assortments))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.visitor_type = self.np_random.integers(3)
        self.utilities = self.TYPE_UTILITIES[self.visitor_type]
        self.click_counts = np.zeros(self.n_categories)
        self.skip_count = 0
        self.steps_remaining = self.max_steps
        return self._get_obs(), self._get_info()
    

    def _get_obs(self):
        return np.array([*(self.click_counts / self.max_steps),
                         self.skip_count / self.max_steps,
                         self.steps_remaining / self.max_steps,
                         ], dtype=np.float32)
    
    def _get_info(self):
        return {"visitor_type": self.visitor_type,
                "type_name": self.TYPE_NAMES[self.visitor_type]}


    def step(self, action):
        assortment = self.assortments[action]
        # MNL choice over displayed categories + outside option
        utils = np.array([self.utilities[j] for j in assortment])
        exp_utils = np.exp(utils)
        exp_outside = np.exp(self.OUTSIDE_UTILITY)
        denom = exp_utils.sum() + exp_outside
        probs = np.append(exp_utils / denom, exp_outside / denom)
        choice = self.np_random.choice(len(probs), p=probs)
        if choice < len(assortment):
            self.click_counts[assortment[choice]] += 1
        else:
            self.skip_count += 1
        self.steps_remaining -= 1
        terminated = self.steps_remaining == 0
        reward = 0.0
        if terminated:
            top2 = np.argsort(self.utilities)[-2:]
            total_clicks = self.click_counts.sum()
            clicks_on_top2 = self.click_counts[top2].sum()
            match_score = clicks_on_top2 / max(1, total_clicks)
            conversion_prob = expit(3 * match_score - 1.5)
            reward = float(self.np_random.random() < conversion_prob)
        return self._get_obs(), reward, terminated, False, self._get_info()