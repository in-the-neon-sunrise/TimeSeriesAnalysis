from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Hashable, Iterable, List, Tuple

import numpy as np
import pandas as pd

from core.markov.markov_models import MarkovResult


class MarkovService:
    STATE_COLUMN_CANDIDATES = (
        "cluster", "cluster_id", "cluster_label", "state", "state_id", "label"
    )
    ORDER_COLUMN_CANDIDATES = (
        "segment_id", "segment", "order", "index", "time", "timestamp", "date"
    )

    def build_model(
        self,
        source_df: pd.DataFrame,
        order: int = 1,
        normalize: bool = True,
        sequential_only: bool = True,
        min_frequency: int = 1,
    ) -> MarkovResult:
        if order < 1:
            raise ValueError("Порядок цепи Маркова должен быть >= 1.")
        if source_df is None or source_df.empty:
            raise ValueError("Нет данных кластеризации для построения цепи Маркова.")

        sequence = self.extract_state_sequence(source_df, sequential_only=sequential_only)
        if len(sequence) < order + 1:
            raise ValueError(
                f"Недостаточно наблюдений для порядка {order}. Нужно минимум {order + 1}, получено {len(sequence)}."
            )

        counts_map, histories, next_states = self._count_transitions(sequence, order)
        counts_df = self._to_matrix(counts_map, histories, next_states)

        if min_frequency > 1:
            counts_df = counts_df.where(counts_df >= min_frequency, other=0)
            counts_df = counts_df.loc[counts_df.sum(axis=1) > 0, counts_df.sum(axis=0) > 0]

        probs_df = self._normalize(counts_df) if normalize else counts_df.copy()
        transitions_long = self._to_long(counts_df, probs_df)

        summary = self._build_summary(sequence, counts_df, probs_df, order)
        stationary = self._stationary_distribution(probs_df, order=order) if normalize else None

        return MarkovResult(
            order=order,
            state_sequence=sequence,
            unique_states=sorted({s for s in sequence}, key=str),
            transition_counts=counts_df,
            transition_probabilities=probs_df,
            transitions_long_table=transitions_long,
            summary=summary,
            params={
                "order": order,
                "normalize": normalize,
                "sequential_only": sequential_only,
                "min_frequency": min_frequency,
            },
            stationary_distribution=stationary,
        )

    def extract_state_sequence(self, source_df: pd.DataFrame, sequential_only: bool = True) -> List[Hashable]:
        state_col = self._choose_column(source_df.columns, self.STATE_COLUMN_CANDIDATES)
        if state_col is None:
            raise ValueError(
                "Не удалось найти колонку состояний/кластеров. Ожидаются: cluster, cluster_id, state, state_id..."
            )

        df = source_df.copy()
        order_col = self._choose_column(df.columns, self.ORDER_COLUMN_CANDIDATES)
        if order_col is not None:
            df = df.sort_values(order_col, kind="stable")

        sequence_series = df[state_col].dropna()
        if not sequential_only and order_col is not None:
            dedup = df[[order_col, state_col]].dropna().drop_duplicates(subset=[order_col], keep="first")
            dedup = dedup.sort_values(order_col, kind="stable")
            sequence_series = dedup[state_col]

        sequence = sequence_series.tolist()
        if not sequence:
            raise ValueError("Последовательность состояний пуста после удаления пропусков.")
        return sequence

    def _count_transitions(
        self,
        sequence: List[Hashable],
        order: int,
    ) -> Tuple[Dict[Tuple[Any, ...], Counter], List[Tuple[Any, ...]], List[Hashable]]:
        counts_map: Dict[Tuple[Any, ...], Counter] = defaultdict(Counter)
        observed_next_states = set()

        for i in range(order, len(sequence)):
            history = tuple(sequence[i - order:i])
            next_state = sequence[i]
            counts_map[history][next_state] += 1
            observed_next_states.add(next_state)

        histories = sorted(counts_map.keys(), key=lambda x: str(x))
        next_states = sorted(observed_next_states, key=str)
        return counts_map, histories, next_states

    def _to_matrix(
        self,
        counts_map: Dict[Tuple[Any, ...], Counter],
        histories: Iterable[Tuple[Any, ...]],
        next_states: Iterable[Hashable],
    ) -> pd.DataFrame:
        history_labels = [self._format_history(h) for h in histories]
        next_labels = [str(s) for s in next_states]

        matrix = pd.DataFrame(0, index=history_labels, columns=next_labels, dtype=float)
        for history in histories:
            history_label = self._format_history(history)
            for next_state, count in counts_map[history].items():
                matrix.at[history_label, str(next_state)] = float(count)
        return matrix

    def _normalize(self, counts_df: pd.DataFrame) -> pd.DataFrame:
        if counts_df.empty:
            return counts_df.copy()
        row_sums = counts_df.sum(axis=1).replace(0, np.nan)
        return counts_df.div(row_sums, axis=0).fillna(0.0)

    def _to_long(self, counts_df: pd.DataFrame, probs_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for history in counts_df.index:
            for next_state in counts_df.columns:
                count = float(counts_df.at[history, next_state])
                if count <= 0:
                    continue
                rows.append(
                    {
                        "history_state": history,
                        "next_state": next_state,
                        "count": count,
                        "probability": float(probs_df.at[history, next_state]) if next_state in probs_df.columns else np.nan,
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["history_state", "next_state", "count", "probability"])
        result = pd.DataFrame(rows)
        return result.sort_values(["probability", "count"], ascending=False).reset_index(drop=True)

    def _build_summary(
        self,
        sequence: List[Hashable],
        counts_df: pd.DataFrame,
        probs_df: pd.DataFrame,
        order: int,
    ) -> Dict[str, Any]:
        total_transitions = int(counts_df.values.sum()) if not counts_df.empty else 0
        n_rows, n_cols = counts_df.shape
        non_zero = int((counts_df.values > 0).sum()) if total_transitions > 0 else 0
        sparsity = 1.0 - (non_zero / (n_rows * n_cols)) if n_rows and n_cols else 1.0

        entropy_by_history = {}
        if not probs_df.empty:
            for history in probs_df.index:
                probs = probs_df.loc[history].values
                entropy_by_history[history] = self._entropy(probs)

        weighted_entropy = None
        if entropy_by_history and total_transitions > 0:
            row_totals = counts_df.sum(axis=1)
            weighted_entropy = float(
                sum(entropy_by_history[h] * row_totals[h] for h in counts_df.index) / max(total_transitions, 1)
            )

        return {
            "order": order,
            "sequence_length": len(sequence),
            "unique_state_count": len(set(sequence)),
            "observed_transitions": total_transitions,
            "matrix_rows": n_rows,
            "matrix_cols": n_cols,
            "non_zero_cells": non_zero,
            "sparsity": float(sparsity),
            "mean_outgoing_entropy": float(np.mean(list(entropy_by_history.values()))) if entropy_by_history else 0.0,
            "weighted_entropy": weighted_entropy if weighted_entropy is not None else 0.0,
            "single_state_only": len(set(sequence)) == 1,
        }

    def _stationary_distribution(self, probs_df: pd.DataFrame, order: int) -> Dict[str, float] | None:
        if order != 1 or probs_df.empty or probs_df.shape[0] != probs_df.shape[1]:
            return None

        matrix = probs_df.to_numpy(dtype=float)
        try:
            vals, vecs = np.linalg.eig(matrix.T)
            idx = int(np.argmin(np.abs(vals - 1)))
            vec = np.real(vecs[:, idx])
            vec = np.clip(vec, 0, None)
            if vec.sum() == 0:
                return None
            vec = vec / vec.sum()
            return {str(state): float(prob) for state, prob in zip(probs_df.index, vec)}
        except Exception:
            return None

    @staticmethod
    def _format_history(history: Tuple[Any, ...]) -> str:
        if len(history) == 1:
            return str(history[0])
        return " | ".join(str(v) for v in history)

    @staticmethod
    def _entropy(probabilities: np.ndarray) -> float:
        probs = probabilities[probabilities > 0]
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def _choose_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
        cols = list(columns)

        for candidate in candidates:
            if candidate in cols:
                return candidate

        lowered = {c.lower(): c for c in cols}
        for candidate in candidates:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()]

        for c in cols:
            c_low = c.lower()
            if any(token in c_low for token in ("cluster", "state", "label")):
                return c
        return None
