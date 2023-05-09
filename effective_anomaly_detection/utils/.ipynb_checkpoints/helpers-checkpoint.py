import numpy as np

def generate_syntehtic_timeseries(n_periods: int,):
    np.random.seed(42)
    x = np.linspace(0, 10, n_periods)
    y = np.sin(x) + np.random.normal(0, 0.6, n_periods) + 100
    return y