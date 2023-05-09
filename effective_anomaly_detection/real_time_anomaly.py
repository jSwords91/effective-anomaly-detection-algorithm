from typing import List, Tuple, Dict
import numpy as np

from .base_anomaly import AnomalyDetector

class RealTimeAnomalyDetector(AnomalyDetector):
    
    """
    Anomaly detector for real-time data streams based on moving average and standard deviation filter.
    """
    
    def __init__(self, lag: int, threshold: float, influence: float):
        super().__init__(lag, threshold, influence)
        self.signals: List[int] = []
        self.filtered_y: List[float] = []
        self.avg_filter: List[float] = []
        self.std_filter: List[float] = []

    def run(self, data: float) -> int:
        self.filtered_y.append(data)
        
        if len(self.filtered_y) < self.lag:
            self.signals.append(0)
        else:
            if len(self.avg_filter) == 0:
                avg, std = self.moving_average_std(self.filtered_y[:self.lag], self.lag)
                self.avg_filter.append(avg)
                self.std_filter.append(std)
            
            is_outlier = abs(data - self.avg_filter[-1]) > self.threshold * self.std_filter[-1]
            signal = np.sign(data - self.avg_filter[-1]) * is_outlier
            self.signals.append(signal)

            filtered_data = self.update_filtered_series(data, self.influence, self.filtered_y[-2]) if is_outlier else data
            self.filtered_y[-1] = filtered_data

            avg, std = self.moving_average_std(self.filtered_y[-self.lag:], self.lag)
            self.avg_filter.append(avg)
            self.std_filter.append(std)

        return self.signals[-1]