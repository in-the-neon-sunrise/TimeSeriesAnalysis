from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Hashable, Iterable, List, Tuple

import numpy as np
import pandas as pd

from core.markov.markov_models import MarkovResult


class MarkovService:
    STATE_COLUMN_CANDIDATES = ("cluster_id", "cluster", "label", "state", "cluster_label", "state_id")
    ORDER_COLUMN_CANDIDATES = (
        "start_idx",
        "start_time",
        "segment_id",
        "end_idx",
        "end_time",
        "timestamp",
        "time",
        "date",
        "segment",
        "order",
        "index",
    )

    def build_model(
        self,
        source_df: pd.DataFrame,
        order: int = 1,
        normalize: bool = True,
        sequential_only: bool = True,
        min_frequency: int = 1,
        progress_callback=None,
        is_cancelled=None,
    ) -> MarkovResult:
        self._check_cancel(is_cancelled)

        if order < 1:
            raise ValueError("Порядок цепи Маркова должен быть >= 1.")
        if source_df is None or source_df.empty:
            raise ValueError("Нет данных кластеризации для построения цепи Маркова.")

        sequence = self.extract_state_sequence(source_df, sequential_only=sequential_only)
        if len(sequence) < order + 1:
            raise ValueError(
                f"Недостаточно наблюдений для порядка {order}. Нужно минимум {order + 1}, получено {len(sequence)}."
            )

        if len(set(sequence)) < 2:
            raise ValueError(
                "В последовательности найдено только одно состояние. "
                "Цепь Маркова можно построить только после кластеризации хотя бы на 2 состояния."
            )

        if progress_callback:
            progress_callback.emit(15, "Последовательность состояний подготовлена")

        counts_map, histories, next_states = self._count_transitions(sequence, order, progress_callback, is_cancelled)
        counts_df = self._to_matrix(counts_map, histories, next_states)
        self._check_cancel(is_cancelled)

        if min_frequency > 1 and not counts_df.empty:
            counts_df = counts_df.where(counts_df >= min_frequency, other=0)
            counts_df = counts_df.loc[counts_df.sum(axis=1) > 0, counts_df.sum(axis=0) > 0]

        if counts_df.empty:
            raise ValueError(
                "После применения минимальной частоты не осталось переходов. "
                "Уменьшите параметр min_frequency."
            )

        probs_df = self._normalize(counts_df) if normalize else counts_df.copy()

        if progress_callback:
            progress_callback.emit(70, "Матрицы переходов построены")

        # БАГ В СТАРОЙ ВЕРСИИ БЫЛ ЗДЕСЬ:
        # transitions_long = self._to_long(counts_df, probs_df, is_cancelled)
        # _to_long принимает только counts_df и probs_df.
        transitions_long = self._to_long(counts_df, probs_df)

        summary = self._build_summary(sequence, counts_df, probs_df, order)
        summary["warnings"] = self._build_warnings(sequence, counts_df, probs_df, order)

        stationary = self._stationary_distribution(probs_df, order=order) if normalize else None

        if progress_callback:
            progress_callback.emit(100, "Markov-модель построена")

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
                "Для построения цепи Маркова нужен результат кластеризации сегментов со столбцом cluster_id. "
                "Сначала выполните кластеризацию сегментов."
            )

        df = source_df.copy()
        sort_columns = self._choose_order_columns(df)
        if sort_columns:
            df = df.sort_values(sort_columns, kind="stable")

        sequence_series = df[state_col].dropna()

        # sequential_only=True означает: берем последовательность сегментов как есть после сортировки.
        # sequential_only=False оставлен для случая, когда в данных есть повторяющиеся временные ключи
        # и нужно удалить дубли по первому order-столбцу.
        if not sequential_only and sort_columns:
            primary_order_col = sort_columns[0]
            dedup = df[[primary_order_col, state_col]].dropna().drop_duplicates(
                subset=[primary_order_col],
                keep="first",
            )
            dedup = dedup.sort_values(primary_order_col, kind="stable")
            sequence_series = dedup[state_col]

        sequence = sequence_series.tolist()
        if not sequence:
            raise ValueError("Последовательность состояний пуста после удаления пропусков.")

        return sequence

    def build_state_sequence_table(self, source_df: pd.DataFrame, sequential_only: bool = True) -> pd.DataFrame:
        state_col = self._choose_column(source_df.columns, self.STATE_COLUMN_CANDIDATES)
        if state_col is None:
            return pd.DataFrame(columns=["position", "state"])

        df = source_df.copy()
        sort_columns = self._choose_order_columns(df)
        if sort_columns:
            df = df.sort_values(sort_columns, kind="stable")

        cols = [c for c in ["segment_id", "start_idx", "end_idx", "start_time", "end_time"] if c in df.columns]
        cols.append(state_col)
        out = df[cols].dropna(subset=[state_col]).copy()
        out = out.rename(columns={state_col: "state"})
        out.insert(0, "position", range(len(out)))
        return out.reset_index(drop=True)

    def build_state_counts_table(self, sequence: List[Hashable]) -> pd.DataFrame:
        if not sequence:
            return pd.DataFrame(columns=["state", "count", "share"])

        counts = pd.Series(sequence).value_counts().sort_index()
        total = len(sequence)
        return pd.DataFrame(
            {
                "state": [str(v) for v in counts.index],
                "count": counts.values.astype(int),
                "share": [float(v / total) for v in counts.values],
            }
        )

    def _count_transitions(
        self,
        sequence: List[Hashable],
        order: int,
        progress_callback=None,
        is_cancelled=None,
    ) -> Tuple[Dict[Tuple[Any, ...], Counter], List[Tuple[Any, ...]], List[Hashable]]:
        counts_map: Dict[Tuple[Any, ...], Counter] = defaultdict(Counter)
        observed_next_states = set()
        total = max(1, len(sequence) - order)

        for idx, i in enumerate(range(order, len(sequence)), start=1):
            self._check_cancel(is_cancelled)

            history = tuple(sequence[i - order:i])
            next_state = sequence[i]
            counts_map[history][next_state] += 1
            observed_next_states.add(next_state)

            if progress_callback and (idx % 100 == 0 or idx == total):
                progress_callback.emit(15 + int((idx / total) * 45), f"Подсчет переходов: {idx}/{total}")

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
                        "probability": float(probs_df.at[history, next_state])
                        if history in probs_df.index and next_state in probs_df.columns
                        else np.nan,
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

        weighted_entropy = 0.0
        if entropy_by_history and total_transitions > 0:
            row_totals = counts_df.sum(axis=1)
            weighted_entropy = float(
                sum(entropy_by_history[h] * row_totals[h] for h in counts_df.index) / max(total_transitions, 1)
            )

        state_counts = Counter(sequence)
        most_common_state, most_common_count = state_counts.most_common(1)[0]

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
            "weighted_entropy": weighted_entropy,
            "single_state_only": len(set(sequence)) == 1,
            "most_common_state": str(most_common_state),
            "most_common_state_share": float(most_common_count / len(sequence)),
        }

    def _build_warnings(
        self,
        sequence: List[Hashable],
        counts_df: pd.DataFrame,
        probs_df: pd.DataFrame,
        order: int,
    ) -> List[str]:
        warnings: List[str] = []

        unique_states = len(set(sequence))
        if len(sequence) < 6:
            warnings.append("Последовательность состояний очень короткая. Вероятности переходов могут быть нестабильными.")

        if order > 1 and counts_df.shape[0] >= len(sequence) - order:
            warnings.append(
                "Для выбранного порядка почти каждая история встречается один раз. "
                "Модель высокого порядка может быть плохо интерпретируемой."
            )

        if unique_states < 3:
            warnings.append(
                "Состояний мало. Матрица переходов будет простой; для более содержательной модели можно проверить кластеризацию."
            )

        state_counts = Counter(sequence)
        _, most_common_count = state_counts.most_common(1)[0]
        if most_common_count / len(sequence) > 0.8:
            warnings.append(
                "Одно состояние занимает более 80% последовательности. "
                "Переходная модель может отражать доминирование одного кластера."
            )

        if not probs_df.empty:
            deterministic_rows = int((probs_df.max(axis=1) >= 0.95).sum())
            if deterministic_rows and deterministic_rows == len(probs_df):
                warnings.append("Почти все переходы детерминированы. Это нормально для коротких последовательностей, но стоит проверить устойчивость.")

        return warnings

    def _stationary_distribution(self, probs_df: pd.DataFrame, order: int) -> Dict[str, float] | None:
        # Стационарное распределение в этом виде имеет простой смысл только для цепи 1-го порядка
        # и квадратной матрицы state -> state.
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

        for col in cols:
            c_low = col.lower()
            if any(token in c_low for token in ("cluster", "state", "label")):
                return col

        return None

    @classmethod
    def _choose_order_columns(cls, df: pd.DataFrame) -> List[str]:
        cols = list(df.columns)
        lower_to_original = {c.lower(): c for c in cols}

        selected: List[str] = []
        for key in cls.ORDER_COLUMN_CANDIDATES:
            col = lower_to_original.get(key)
            if col is not None and col not in selected:
                selected.append(col)

        if selected:
            return selected

        fallback = cls._choose_column(cols, cls.ORDER_COLUMN_CANDIDATES)
        return [fallback] if fallback else []

    @staticmethod
    def _check_cancel(is_cancelled):
        if is_cancelled and is_cancelled():
            raise RuntimeError("Задача отменена")
