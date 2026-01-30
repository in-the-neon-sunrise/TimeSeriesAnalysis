import pandas as pd

def build_data_profile(df: pd.DataFrame):
    profile = []

    for column in df.columns:
        series = df[column]

        info = {
            "column": column,
            "dtype": str(series.dtype),
            "non_null": series.count(),
            "nulls": series.isna().sum(),
        }

        if pd.api.types.is_numeric_dtype(series):
            info.update({
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
            })
        else:
            info.update({
                "min": None,
                "max": None,
                "mean": None,
            })

        profile.append(info)

    return profile