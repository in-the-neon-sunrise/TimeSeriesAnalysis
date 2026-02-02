from dataclasses import dataclass, field
from typing import Optional, Dict
import pandas as pd


@dataclass
class ProjectState:
    # данные
    raw_data: Optional[pd.DataFrame] = None
    preprocessed_data: Optional[pd.DataFrame] = None
    features: Optional[pd.DataFrame] = None
    segments: Optional[pd.DataFrame] = None
    clusters: Optional[pd.DataFrame] = None
    markov_matrix: Optional[pd.DataFrame] = None

    # параметры анализа
    params: Dict = field(default_factory=dict)

    # служебное
    file_path: Optional[str] = None
    is_dirty: bool = False