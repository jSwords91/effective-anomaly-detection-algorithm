from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np

@dataclass
class AnomalyDetectorConfig:
    lag: int
    threshold: float
    influence: float

class AnomalyDetector:
    def __init__(self, config: AnomalyDetectorConfig):
        self.lag = config.lag
        self.threshold = config.threshold
        self.influence = config.influence

    @staticmethod
    def moving_average_std(data: List[float], window_size: int) -> Tuple[float, float]:
        return np.mean(data[-window_size:]), np.std(data[-window_size:])

    @staticmethod
    def update_filtered_series(data: float, influence: float, previous_filtered_value: float) -> float:
        return influence * data + (1 - influence) * previous_filtered_value

    @staticmethod
    def update_avg_std(filtered_y: List[float], avg_filter: List[float], std_filter: List[float], i: int, lag: int) -> Tuple[List[float], List[float]]:
        avg_filter[i], std_filter[i] = AnomalyDetector.moving_average_std(filtered_y[i - lag + 1: i + 1], lag)
        return avg_filter, std_filter

    @staticmethod
    def create_arrays(y: List[float]) -> Tuple[np.ndarray, List[float], List[float]]:
        return np.zeros(len(y)), [0] * len(y), [0] * len(y)
    
    @staticmethod
    def is_outlier(data: float, avg: float, std: float, threshold: float) -> bool:
        return abs(data - avg) > threshold * std
    

