from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class ClusteringResult:
    method: str
    params: Dict[str, Any]
    selected_columns: List[str]
    labels: List[int]
    clustered_segments: pd.DataFrame
    metrics: Dict[str, float | None]
    summary: Dict[str, Any]
    distance_metric: str
    source_info: Dict[str, Any] = field(default_factory=dict)

    def to_project_payload(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "params": dict(self.params),
            "selected_columns": list(self.selected_columns),
            "labels": list(self.labels),
            "clustered_segments": self.clustered_segments.copy(),
            "metrics": dict(self.metrics),
            "summary": dict(self.summary),
            "distance_metric": self.distance_metric,
            "source_info": dict(self.source_info),
        }
