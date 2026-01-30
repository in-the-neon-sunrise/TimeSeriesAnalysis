from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

@dataclass
class Project:
    name: str = "Untitled project"
    csv_path: Optional[str] = None

    raw_data: Optional[pd.DataFrame] = None
    processed_data: Optional[pd.DataFrame] = None

    features: Optional[pd.DataFrame] = None
    segments: Optional[pd.DataFrame] = None
    clusters: Optional[pd.DataFrame] = None

    markov_matrix: Optional[pd.DataFrame] = None