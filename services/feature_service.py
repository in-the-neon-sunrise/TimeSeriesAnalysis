import numpy as np
import pandas as pd


class FeatureService:

    @staticmethod
    def extract_features(series, window_size, step, features, progress_callback=None, is_cancelled=None):

        values = series.values
        n = len(values)
        rows = []
        total = max(1, len(range(0, n - window_size + 1, step)))

        for idx, start in enumerate(range(0, n - window_size + 1, step), start=1):
            if is_cancelled and is_cancelled():
                raise RuntimeError("Задача отменена")

            window = values[start:start + window_size]
            row = {}

            if "mean" in features:
                row["mean"] = np.mean(window)

            if "std" in features:
                row["std"] = np.std(window)

            if "var" in features:
                row["var"] = np.var(window)

            if "min" in features:
                row["min"] = np.min(window)

            if "max" in features:
                row["max"] = np.max(window)

            if "skew" in features:
                row["skew"] = pd.Series(window).skew()

            if "kurt" in features:
                row["kurt"] = pd.Series(window).kurt()

            if "diff" in features:
                diffs = np.diff(window)
                row["diff_mean"] = np.mean(diffs)

            if "gradient" in features:
                grad = np.gradient(window)
                row["gradient_mean"] = np.mean(grad)

            if "roc" in features:
                roc = np.diff(window) / (window[:-1] + 1e-9)
                row["roc_mean"] = np.mean(roc)

            if "rms" in features:
                row["rms"] = np.sqrt(np.mean(window ** 2))

            if "energy" in features:
                row["energy"] = np.sum(window ** 2)

            if "ptp" in features:
                row["ptp"] = np.ptp(window)

            rows.append(row)

            if progress_callback:
                progress = int((idx / total) * 100)
                progress_callback.emit(progress, f"Обработано окон: {idx}/{total}")

        feature_df = pd.DataFrame(rows)
        return feature_df