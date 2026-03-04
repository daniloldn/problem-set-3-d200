import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src_1.metrics import total_rewards, regret
from src_1.bandit_env import AdChannelBandit


def run_experiment(models, T=2000, n_runs=50, base_seed=42):

    avg_cum_rewards = {}
    avg_cum_regrets = {}
    avg_frequency = {}

    for name, model in models.items():
        all_cum_rewards = np.empty((n_runs, T), dtype=float)
        all_cum_regrets = np.empty((n_runs, T), dtype=float)
        all_frequency = None

        for i in range(n_runs):
            # Fresh bandit each run (avoids RNG state coupling)
            bandit = AdChannelBandit()

            if all_frequency is None:
                all_frequency = np.empty((n_runs, bandit.K), dtype=float)

            results = model(bandit, T, seed=base_seed + i)

            # You already have these helpers
            cum_rewards = total_rewards(results["rewards"])   # shape (T,)
            cum_regrets = regret(bandit, results["arms"])     # shape (T,)

            all_cum_rewards[i, :] = cum_rewards
            all_cum_regrets[i, :] = cum_regrets
            
             # frequency: count pulls per arm, divide by T  -> shape (K,)
            arms = np.asarray(results["arms"], dtype=int)
            counts = np.bincount(arms, minlength=bandit.K)            # (K,)
            freqs = counts / T
            all_frequency[i, :] = freqs


        avg_cum_rewards[name] = all_cum_rewards.mean(axis=0)
        avg_cum_regrets[name] = all_cum_regrets.mean(axis=0)
        avg_frequency[name] = all_frequency.mean(axis=0)

    return avg_cum_rewards, avg_cum_regrets, avg_frequency


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


def plot_freq(avg_frequency, title="Average pull frequency by arm"):
    """
    avg_frequency: dict[name -> np.ndarray (K,)]
    """
    K = len(next(iter(avg_frequency.values())))
    x = np.arange(K)

    fig = go.Figure()
    for name, freqs in avg_frequency.items():
        fig.add_trace(go.Bar(x=x, y=freqs, name=name))

    fig.update_layout(
        title=title,
        xaxis_title="arm",
        yaxis_title="frequency",
        barmode="group",
        legend_title="policy"
    )
    fig.show()

