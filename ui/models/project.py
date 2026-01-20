from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class Project:
    csv_path: str | None = None
    dataframe: pd.DataFrame | None = None

    def has_data(self) -> bool:
        return self.dataframe is not None