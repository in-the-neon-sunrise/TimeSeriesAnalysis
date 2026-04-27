from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional

import pandas as pd


@dataclass
class MarkovResult:
    order: int
    state_sequence: List[Hashable]
    unique_states: List[Hashable]
    transition_counts: pd.DataFrame
    transition_probabilities: pd.DataFrame
    transitions_long_table: pd.DataFrame
    summary: Dict[str, Any]
    params: Dict[str, Any] = field(default_factory=dict)
    stationary_distribution: Optional[Dict[str, float]] = None

    def to_project_payload(self) -> Dict[str, Any]:
        return {
            "order": self.order,
            "state_sequence": [str(v) for v in self.state_sequence],
            "unique_states": [str(v) for v in self.unique_states],
            "transition_counts": self.transition_counts.copy(),
            "transition_probabilities": self.transition_probabilities.copy(),
            "transitions_long_table": self.transitions_long_table.copy(),
            "summary": dict(self.summary),
            "params": dict(self.params),
            "stationary_distribution": self.stationary_distribution,
        }
