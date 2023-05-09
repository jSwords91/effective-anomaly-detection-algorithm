from typing import List, Dict
import numpy as np

from .base_anomaly import AnomalyDetector, AnomalyDetectorConfig

class RealTimeAnomalyDetector(AnomalyDetector):
    
    """
    Anomaly detector for real-time data streams based on moving average and standard deviation filter.
    """
    
    def __init__(self, config: AnomalyDetectorConfig):
        super().__init__(config)
        self.signals: List[int] = []
        self.filtered_y: List[float] = []
        self.avg_filter: List[float] = []
        self.std_filter: List[float] = []

    def run(self, data: float) -> int:
        self.filtered_y.append(data)

        if len(self.filtered_y) < self.lag:
            self.signals.append(0)
        else:
            if len(self.filtered_y) == self.lag:
                avg, std = self.moving_average_std(self.filtered_y[-self.lag:], self.lag)
                self.avg_filter.append(avg)
                self.std_filter.append(std)

            is_outlier = AnomalyDetector.is_outlier(data, self.avg_filter[-1], self.std_filter[-1], self.threshold)
            signal = np.sign(data - self.avg_filter[-1]) * is_outlier
            self.signals.append(signal)

            if is_outlier:
                filtered_data = self.update_filtered_series(data, self.influence, self.filtered_y[-2])
                self.filtered_y[-1] = filtered_data
                avg, std = self.moving_average_std(self.filtered_y[-self.lag:], self.lag)
                self.avg_filter.append(avg)
                self.std_filter.append(std)
            else:
                self.avg_filter.append(self.avg_filter[-1])
                self.std_filter.append(self.std_filter[-1])

        return self.signals[-1]
