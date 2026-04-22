from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from external import SDA


class SDAAdapter:
    """Thin adapter around external SDA implementation."""

    def run(
        self,
        features: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        sda = SDA(**params)
        result, df_st_edges = sda.apply(features)
        return result, df_st_edges
