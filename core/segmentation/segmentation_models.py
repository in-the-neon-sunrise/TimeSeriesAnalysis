from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class SegmentationResult:
    input_kind: str
    selected_columns: List[str]
    params: Dict[str, Any]

    results_table: pd.DataFrame
    stage1_results: pd.DataFrame
    best_result_row: Dict[str, Any]

    edges: List[int]
    stage_ids: List[int]

    segmented_data: pd.DataFrame
    segments_table: pd.DataFrame

    summary: Dict[str, Any] = field(default_factory=dict)
    timestamp_column: Optional[str] = None
