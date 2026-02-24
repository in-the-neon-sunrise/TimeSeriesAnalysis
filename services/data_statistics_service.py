import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from scipy import stats


class DataStatisticsService:

    @staticmethod
    def basic_info(df: pd.DataFrame) -> dict:
        total_rows = len(df)
        total_columns = df.shape[1]
        missing_ratio = df.isna().mean().mean()

        return {
            "rows": total_rows,
            "columns": total_columns,
            "missing_ratio": missing_ratio
        }

    @staticmethod
    def descriptive_statistics(series: pd.Series) -> dict:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        return {
            "mean": series.mean(),
            "median": series.median(),
            "min": series.min(),
            "max": series.max(),
            "std": series.std(),
            "variance": series.var(),
            "q1": q1,
            "q3": q3,
            "iqr": iqr
        }

    @staticmethod
    def stationarity_adf(series: pd.Series) -> dict:
        result = adfuller(series.dropna())

        return {
            "adf_statistic": result[0],
            "p_value": result[1],
            "is_stationary": result[1] < 0.05
        }

    @staticmethod
    def autocorrelation(series: pd.Series, lags: int = 40):
        return acf(series.dropna(), nlags=lags)

    @staticmethod
    def detect_outliers_iqr(series: pd.Series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (series < lower) | (series > upper)

        return {
            "count": mask.sum(),
            "mask": mask
        }

    @staticmethod
    def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0):
        z_scores = np.abs(stats.zscore(series.dropna()))
        mask = z_scores > threshold

        return {
            "count": mask.sum(),
            "mask": mask
        }