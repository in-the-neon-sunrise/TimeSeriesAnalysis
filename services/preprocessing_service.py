import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class PreprocessingService:

    #Масштабирование

    @staticmethod
    def scale(series: pd.Series, method: str) -> pd.Series:
        values = series.values.reshape(-1, 1)

        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "zscore":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            return series

        scaled = scaler.fit_transform(values)
        return pd.Series(scaled.flatten(), index=series.index)

    #Пропуски

    @staticmethod
    def handle_missing(series: pd.Series, method: str) -> pd.Series:
        if method == "drop":
            return series.dropna()
        elif method == "mean":
            return series.fillna(series.mean())
        elif method == "median":
            return series.fillna(series.median())
        elif method == "interpolate":
            return series.interpolate()
        else:
            return series

    #Сглаживание

    @staticmethod
    def smooth(series: pd.Series, method: str, **params) -> pd.Series:
        if method == "moving_average":
            window = params.get("window", 5)
            return series.rolling(window=window).mean()
        elif method == "median":
            window = params.get("window", 5)
            return series.rolling(window=window).median()
        elif method == "ewm":
            alpha = params.get("alpha", 0.3)
            return series.ewm(alpha=alpha).mean()
        else:
            return series