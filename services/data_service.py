import pandas as pd

class DataService:
    def load_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)