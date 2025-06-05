import numpy as np
import typing


class MinMaxScaler:
    def __init__(self):
        self.min_vals = None
        self.max_vals = None

    def fit(self, data: np.ndarray) -> None:
        self.min_vals = np.min(data, axis=0)
        self.max_vals = np.max(data, axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("The scaler hasn't been fitted yet")
        return (data - self.min_vals) / (self.max_vals - self.min_vals)


class StandardScaler:
    def fit(self, data: np.ndarray) -> None:
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0, ddof=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not hasattr(self, "mean") or not hasattr(self, "std"):
            raise RuntimeError("The scaler hasn't been fitted yet")
        return (data - self.mean) / self.std
