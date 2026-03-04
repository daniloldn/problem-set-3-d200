import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.metrics import total_rewards, regret
from src.bandit_env import AdChannelBandit


def run_experiment(models, T=2000, n_runs=50, base_seed=42):

    avg_cum_rewards = {}
    avg_cum_regrets = {}

    for name, model in models.items():
        all_cum_rewards = np.empty((n_runs, T), dtype=float)
        all_cum_regrets = np.empty((n_runs, T), dtype=float)

        for i in range(n_runs):
            # Fresh bandit each run (avoids RNG state coupling)
            bandit = AdChannelBandit()

            results = model(bandit, T, seed=base_seed + i)

            # You already have these helpers
            cum_rewards = total_rewards(results["rewards"])   # shape (T,)
            cum_regrets = regret(bandit, results["arms"])     # shape (T,)

            all_cum_rewards[i, :] = cum_rewards
            all_cum_regrets[i, :] = cum_regrets

        avg_cum_rewards[name] = all_cum_rewards.mean(axis=0)
        avg_cum_regrets[name] = all_cum_regrets.mean(axis=0)

    return avg_cum_rewards, avg_cum_regrets


def plot_avg_curves(avg_curves, title, y_label):
    """
    Plot multiple average curves on one figure.
    avg_curves: dict[name -> np.ndarray (T,)]
    """
    T = len(next(iter(avg_curves.values())))
    x = np.arange(T)

    fig = go.Figure()
    for name, y in avg_curves.items():
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))

    fig.update_layout(
        title=title,
        xaxis_title="t",
        yaxis_title=y_label,
        legend_title="policy"
    )
    fig.show()

