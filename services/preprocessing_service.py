import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


class PreprocessingService:

    #Масштабирование

    @staticmethod
    def scale(series: pd.Series, method: str) -> pd.Series:
        if method == "none":
            return series

        if series.empty:
            return series

        clean = series.replace([np.inf, -np.inf], np.nan)
        if clean.dropna().empty:
            return clean

        values = clean.values.reshape(-1, 1)

        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "zscore":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            return series

        scaled = scaler.fit_transform(values)
        return pd.Series(scaled.flatten(), index=clean.index, name=clean.name)

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
            return series.rolling(window=window, min_periods=1).mean()
        elif method == "median":
            window = params.get("window", 5)
            return series.rolling(window=window, min_periods=1).median()
        elif method == "ewm":
            alpha = params.get("alpha", 0.3)
            return series.ewm(alpha=alpha).mean()
        else:
            return series

    @staticmethod
    def apply_pipeline(
        series: pd.Series,
        missing_method: str = "none",
        smoothing_method: str = "none",
        scaling_method: str = "none",
        window: int = 5,
        alpha: float = 0.3
    ) -> pd.Series:
        processed = series.copy()
        processed = PreprocessingService.handle_missing(processed, missing_method)
        processed = PreprocessingService.smooth(
            processed,
            smoothing_method,
            window=window,
            alpha=alpha,
        )
        processed = PreprocessingService.scale(processed, scaling_method)
        return processed

    @staticmethod
    def series_summary(series: pd.Series) -> dict:
        clean = series.replace([np.inf, -np.inf], np.nan)
        valid = clean.dropna()

        summary = {
            "rows": int(len(clean)),
            "missing": int(clean.isna().sum()),
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }

        if valid.empty:
            return summary

        summary.update(
            {
                "mean": float(valid.mean()),
                "median": float(valid.median()),
                "std": float(valid.std()),
                "min": float(valid.min()),
                "max": float(valid.max()),
                "q25": float(valid.quantile(0.25)),
                "q75": float(valid.quantile(0.75)),
            }
        )
        return summary

    @staticmethod
    def build_preview(original: pd.Series, processed: pd.Series, n: int = 10) -> pd.DataFrame:
        head_processed = processed.head(n)
        preview = pd.DataFrame(
            {
                "index": head_processed.index,
                "original": original.reindex(head_processed.index).values,
                "processed": head_processed.values,
            }
        )
        return preview