from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .base_anomaly import AnomalyDetector, AnomalyDetectorConfig

class EMAAnomalyDetector(AnomalyDetector):
    
    """
    Anomaly detector based on thresholding technique using Exponential Moving Average (EMA) and standard deviation filter.
    """

    def __init__(self, config: AnomalyDetectorConfig, alpha: float = 0.1):
        super().__init__(config)
        self.alpha = alpha

    @staticmethod
    def exponential_moving_average_std(data: List[float], alpha: float) -> Tuple[float, float]:
        ema = data[0]
        for value in data[1:]:
            ema = alpha * value + (1 - alpha) * ema
        std = np.std(data)
        return ema, std

    def run(self, y: List[float]) -> Dict[str, np.ndarray]:
        y = np.asarray(y)
        signals, avg_filter, std_filter = self.create_arrays(y)
        filtered_y = np.array(y)
        avg_filter[self.lag - 1], std_filter[self.lag - 1] = self.exponential_moving_average_std(y[:self.lag], self.alpha)

        for i, data in enumerate(y[self.lag:], start=self.lag):
            is_outlier = self.is_outlier(data, avg_filter[i - 1], std_filter[i - 1], self.threshold)
            signals[i] = np.sign(data - avg_filter[i - 1]) * is_outlier

            filtered_y[i] = self.update_filtered_series(data, self.influence, filtered_y[i - 1]) if is_outlier else data
            avg_filter[i], std_filter[i] = self.exponential_moving_average_std(filtered_y[i - self.lag + 1: i + 1], self.alpha)

        return dict(signals=np.asarray(signals),
                    avgFilter=np.asarray(avg_filter),
                    stdFilter=np.asarray(std_filter))

    
    def plot(self, y: List[float], results: Dict[str, np.ndarray], xlabel: str = 'Time', ylabel: str = 'Value', ylabel2: str = 'Signal', figsize: Tuple[int, int] = (12, 8)) -> plt.figure:
        
        fig = plt.figure(figsize=figsize, dpi=150)
        gs = gridspec.GridSpec(3, 1, height_ratios=[2, 0.5, 1])
        gs.update(wspace=1.5, hspace=0.1)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)

        signals = results["signals"][self.lag:]
        avg_filter = results["avgFilter"][self.lag:]
        std_filter = results["stdFilter"][self.lag:]
        time_series = np.arange(len(y))[self.lag:]
        y = y[self.lag:]
        upper_bound = (avg_filter + self.threshold * std_filter)
        lower_bound = (avg_filter - self.threshold * std_filter)

        ax1.plot(time_series, y, 'k.', label='Original Data', alpha=0.7)
        ax1.plot(time_series, avg_filter, ls='-', lw=2, c='steelblue', label='Moving Average')
        ax1.fill_between(time_series, lower_bound, upper_bound, color='lightsteelblue', alpha=0.3, label='Bounds')
        ax1.scatter(time_series[signals == 1], y[signals == 1], color='coral', s=20, zorder=5)
        ax1.scatter(time_series[signals == -1], y[signals == -1], color='coral', marker='o', s=20, zorder=5)
        ax1.grid(True, which='major', c='gray', ls='-', lw=0.5, alpha=0.1)
        ax2.set_xlabel(xlabel, fontfamily="monospace")
        ax1.set_ylabel(ylabel, fontfamily="monospace")
        ax2.plot(time_series, signals, ls='-', c='coral', label='Signals')
        ax2.set_ylabel(ylabel2, fontfamily="monospace")
        ax2.grid(True, which='major', c='gray', ls='-', lw=0.5, alpha=0.1)

        for s in ["bottom", "top", "left", "right"]:
            ax1.spines[s].set_visible(False)
            ax2.spines[s].set_visible(False)

        ax1.tick_params(axis=u'both', which=u'both', length=0)
        ax2.tick_params(axis=u'both', which=u'both', length=0)
        ax1.set_xlim(time_series[0], time_series[-1])
        ax1.set_ylim(lower_bound.min(), upper_bound.max())
        ax2.set_xlim(time_series[0], time_series[-1])
        ax1.set_xticklabels([])

        return fig
    

