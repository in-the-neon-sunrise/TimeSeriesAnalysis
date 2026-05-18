from __future__ import annotations

import ast
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from core.segmentation.sda_adapter import SDAAdapter
from core.segmentation.segmentation_models import SegmentationResult


class SegmentationService:

    def __init__(self, adapter: Optional[SDAAdapter] = None):
        self.adapter = adapter or SDAAdapter()

    def run_segmentation(
        self,
        features_df: pd.DataFrame,
        selected_columns: List[str],
        params: Dict[str, Any],
        input_kind: str = "features",
        timestamp_series: Optional[pd.Series] = None,
        progress_callback=None,
        is_cancelled=None,
    ) -> SegmentationResult:
        self._check_cancel(is_cancelled)

        if features_df is None or features_df.empty:
            raise ValueError("Нет данных для сегментации.")

        selected_columns = selected_columns or list(features_df.select_dtypes(include="number").columns)
        mode = str(params.get("mode", "full")).lower()
        if mode not in {"full", "chunked"}:
            raise ValueError("Некорректный режим сегментации. Доступно: full/chunked.")

        data_for_sda, used_columns, prepare_warnings = self._prepare_numeric_data(features_df, selected_columns)
        if data_for_sda.shape[0] < 3:
            raise ValueError("Для сегментации нужно минимум 3 строки данных.")
        if self._has_only_constant_columns(data_for_sda):
            raise ValueError("Все выбранные признаки константные. Сегментация невозможна.")

        data_quality = self._data_quality_diagnostics(data_for_sda)
        warnings = prepare_warnings + data_quality["warnings"]

        sda_params = self._extract_sda_params(params)
        sda_params = self._sanitize_sda_params(sda_params, n_rows=len(data_for_sda))

        # Масштабируем здесь, чтобы не зависеть от внутреннего scale SDA.
        # В параметры SDA передаем scale=False, иначе масштабирование может выполниться дважды.
        features_for_algo = self._scale_if_needed(data_for_sda.values, bool(params.get("scale", False)))
        sda_params["scale"] = False

        if progress_callback:
            progress_callback.emit(15, "Подготовка данных завершена")
        self._check_cancel(is_cancelled)

        if mode == "chunked":
            result_payload = self.run_chunked_segmentation(
                data=data_for_sda,
                selected_columns=used_columns,
                params=sda_params,
                original_params=params,
                chunk_size=int(params.get("chunk_size", 1000)),
                overlap=int(params.get("overlap", 100)),
                min_segment_len=int(params.get("min_segment_len", 20)),
                merge_boundaries_tolerance=int(params.get("merge_boundaries_tolerance", 10)),
                min_score_to_split=float(params.get("min_score_to_split", 0.03)),
                progress_callback=progress_callback,
                is_cancelled=is_cancelled,
            )
            results_table = result_payload["results_table"]
            stage1_results = result_payload["stage1_results"]
            edges = result_payload["edges"]
            best_row = result_payload["best_row"]
            chunk_stats = result_payload["chunk_stats"]
            warnings.extend(result_payload.get("warnings", []))
        else:
            results_table, stage1_results = self.adapter.run(features_for_algo, sda_params)
            if results_table is None or results_table.empty:
                raise ValueError("SDA вернул пустой результат.")

            if progress_callback:
                progress_callback.emit(70, "SDA завершен, формирование результатов")
            self._check_cancel(is_cancelled)

            results_table = self._augment_results_table(results_table, len(data_for_sda))
            best_row = self._select_best_result(results_table, len(data_for_sda), params)
            edges = self._extract_edges(best_row.get("St_edges"), len(data_for_sda))
            chunk_stats = []

        stage_ids = self._edges_to_stage_ids(len(data_for_sda), edges)

        segmented_data = self._build_segmented_data(
            data_for_sda,
            stage_ids,
            edges,
            timestamp_series=timestamp_series,
        )
        segments_table = self._build_segments_table(segmented_data, used_columns)
        summary = self._build_summary(results_table, best_row, edges, segments_table, params)

        if chunk_stats:
            summary["chunk_stats"] = chunk_stats
            summary["mode"] = "chunked"
            summary["chunks_processed"] = len(chunk_stats)
            summary["chunk_count_estimate"] = params.get("chunk_count_estimate", len(chunk_stats))
        else:
            summary["mode"] = "full"
            summary["chunk_count_estimate"] = params.get("chunk_count_estimate")

        warnings.extend(self._result_warnings(summary, params, segments_table))
        summary["warnings"] = list(dict.fromkeys([w for w in warnings if w]))

        if progress_callback:
            progress_callback.emit(100, "Сегментация завершена")

        return SegmentationResult(
            input_kind=input_kind,
            selected_columns=used_columns,
            params=params,
            results_table=results_table.reset_index(drop=True),
            stage1_results=stage1_results.reset_index(drop=True) if stage1_results is not None else pd.DataFrame(),
            best_result_row=best_row,
            edges=edges,
            stage_ids=stage_ids,
            segmented_data=segmented_data,
            segments_table=segments_table,
            summary=summary,
            timestamp_column=timestamp_series.name if timestamp_series is not None else None,
        )

    # =========================
    # Chunked segmentation
    # =========================

    def run_chunked_segmentation(
        self,
        data: pd.DataFrame,
        selected_columns: List[str],
        params: Dict[str, Any],
        original_params: Dict[str, Any],
        chunk_size: int,
        overlap: int,
        min_segment_len: int,
        merge_boundaries_tolerance: int,
        min_score_to_split: float,
        progress_callback=None,
        is_cancelled=None,
    ) -> Dict[str, Any]:
        n_rows = len(data)
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "Некорректные параметры фрагментации: требуется chunk_size > 0 и 0 <= overlap < chunk_size."
            )

        chunk_size = min(chunk_size, n_rows)
        step = chunk_size - overlap
        if step <= 0:
            raise ValueError("Некорректные параметры фрагментации: шаг должен быть больше 0.")

        global_edges: List[int] = [0, n_rows]
        all_results: List[pd.DataFrame] = []
        all_stage1: List[pd.DataFrame] = []
        chunk_stats: List[Dict[str, Any]] = []
        warnings: List[str] = []

        starts = list(range(0, n_rows, step))
        total_chunks = len(starts)

        for i, start in enumerate(starts, start=1):
            self._check_cancel(is_cancelled)

            end = min(start + chunk_size, n_rows)
            chunk = data.iloc[start:end]

            if len(chunk) < 3:
                chunk_stats.append({"chunk_id": i, "start": start, "end": end, "status": "skipped_too_small"})
                continue

            if self._has_only_constant_columns(chunk):
                chunk_stats.append({"chunk_id": i, "start": start, "end": end, "status": "skipped_constant"})
                continue

            try:
                chunk_params = self._sanitize_sda_params(dict(params), n_rows=len(chunk))
                chunk_features = self._scale_if_needed(chunk.values, bool(original_params.get("scale", False)))
                chunk_params["scale"] = False

                result_tbl, st1_tbl = self.adapter.run(chunk_features, chunk_params)
                if result_tbl is None or result_tbl.empty:
                    chunk_stats.append({"chunk_id": i, "start": start, "end": end, "status": "skipped_empty_result"})
                    continue

                result_tbl = self._augment_results_table(result_tbl, len(chunk))
                best_row = self._select_best_result(result_tbl, len(chunk), original_params)

                score = self._quality_score(best_row)
                local_edges = self._extract_edges(best_row.get("St_edges"), len(chunk))
                local_segments = len(local_edges) + 1

                if score >= min_score_to_split and local_edges:
                    global_edges.extend([start + edge for edge in local_edges if 0 < edge < len(chunk)])
                    status = "accepted"
                elif not local_edges:
                    status = "no_edges_skipped"
                else:
                    status = "low_score_skipped"

                chunk_stats.append({
                    "chunk_id": i,
                    "start": start,
                    "end": end,
                    "length": len(chunk),
                    "score": score,
                    "local_segments": local_segments,
                    "local_edges": local_edges,
                    "status": status,
                })

                result_tbl = result_tbl.copy()
                result_tbl["chunk_id"] = i
                result_tbl["chunk_start"] = start
                result_tbl["chunk_end"] = end
                all_results.append(result_tbl)

                if st1_tbl is not None and not st1_tbl.empty:
                    st1_tbl = st1_tbl.copy()
                    st1_tbl["chunk_id"] = i
                    st1_tbl["chunk_start"] = start
                    st1_tbl["chunk_end"] = end
                    all_stage1.append(st1_tbl)

            except Exception as exc:
                chunk_stats.append({
                    "chunk_id": i,
                    "start": start,
                    "end": end,
                    "status": "failed",
                    "error": str(exc),
                })

            if progress_callback:
                progress = int(15 + (i / total_chunks) * 75)
                progress_callback.emit(progress, f"Обработка части {i} из {total_chunks}")

        if not all_results:
            raise ValueError("SDA не смог обработать ни одной части данных.")

        accepted = [s for s in chunk_stats if s.get("status") == "accepted"]
        if not accepted:
            warnings.append(
                "Ни одна часть не прошла порог качества разбиения. Итоговая сегментация может состоять из одного сегмента."
            )

        edges = self.normalize_edges(global_edges, n_rows)
        edges = self.merge_close_edges(edges, merge_boundaries_tolerance, n_rows)
        edges = self.remove_short_segments(edges, min_segment_len, n_rows)

        results_table = pd.concat(all_results, ignore_index=True)
        stage1_results = pd.concat(all_stage1, ignore_index=True) if all_stage1 else pd.DataFrame()
        best_row = self._select_best_result(results_table, n_rows, original_params)

        if progress_callback:
            progress_callback.emit(
                90,
                f"Сегментация завершена: обработано {total_chunks} частей, найдено {len(edges) - 1} сегментов",
            )

        return {
            "results_table": results_table,
            "stage1_results": stage1_results,
            "edges": edges[1:-1],
            "best_row": best_row,
            "chunk_stats": chunk_stats,
            "warnings": warnings,
        }

    # =========================
    # Edges postprocessing
    # =========================

    def normalize_edges(self, edges: List[int], n_rows: int) -> List[int]:
        out = []
        for edge in edges:
            try:
                edge_int = int(edge)
            except Exception:
                continue
            if 0 <= edge_int <= n_rows:
                out.append(edge_int)

        out.extend([0, n_rows])
        return sorted(set(out))

    def merge_close_edges(self, edges: List[int], tolerance: int, n_rows: int) -> List[int]:
        edges = self.normalize_edges(edges, n_rows)
        if tolerance <= 0 or len(edges) <= 2:
            return edges

        merged = [edges[0]]
        for edge in edges[1:]:
            if edge in (0, n_rows):
                merged.append(edge)
                continue

            if edge - merged[-1] <= tolerance and merged[-1] not in (0, n_rows):
                merged[-1] = int(round((merged[-1] + edge) / 2))
            else:
                merged.append(edge)

        return self.normalize_edges(merged, n_rows)

    def remove_short_segments(self, edges: List[int], min_segment_len: int, n_rows: int) -> List[int]:
        if min_segment_len <= 1:
            return self.normalize_edges(edges, n_rows)

        normalized = self.normalize_edges(edges, n_rows)
        i = 1
        while i < len(normalized) - 1:
            left, curr, right = normalized[i - 1], normalized[i], normalized[i + 1]
            if (curr - left) < min_segment_len or (right - curr) < min_segment_len:
                del normalized[i]
                continue
            i += 1

        return self.normalize_edges(normalized, n_rows)

    # =========================
    # Data preparation
    # =========================

    def _prepare_numeric_data(
        self,
        df: pd.DataFrame,
        selected_columns: Iterable[str],
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        warnings: List[str] = []

        missing = [c for c in selected_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Выбранные столбцы отсутствуют в данных: {missing}")

        subset = df.loc[:, list(selected_columns)]
        numeric = subset.select_dtypes(include="number")
        dropped_non_numeric = [c for c in subset.columns if c not in numeric.columns]
        if dropped_non_numeric:
            warnings.append(f"Нечисловые столбцы не использовались: {', '.join(map(str, dropped_non_numeric))}.")

        if numeric.empty:
            raise ValueError("После фильтрации не осталось числовых признаков.")

        cleaned = numeric.replace([np.inf, -np.inf], np.nan)
        before_rows = len(cleaned)
        cleaned = cleaned.dropna(axis=0, how="any")
        dropped_rows = before_rows - len(cleaned)
        if dropped_rows:
            warnings.append(f"Удалено строк с NaN/Inf перед SDA: {dropped_rows}.")

        constant_cols = [c for c in cleaned.columns if cleaned[c].nunique(dropna=True) <= 1]
        if constant_cols:
            cleaned = cleaned.drop(columns=constant_cols)
            warnings.append(f"Константные признаки исключены перед SDA: {', '.join(map(str, constant_cols))}.")

        if cleaned.empty:
            raise ValueError("Все строки или признаки были удалены после очистки NaN/Inf и константных столбцов.")

        return cleaned, list(cleaned.columns), warnings

    @staticmethod
    def _has_only_constant_columns(df: pd.DataFrame) -> bool:
        return all(df[c].nunique(dropna=True) <= 1 for c in df.columns)

    def _data_quality_diagnostics(self, df: pd.DataFrame) -> Dict[str, Any]:
        warnings: List[str] = []
        unique_rows = int(df.drop_duplicates().shape[0])
        unique_ratio = unique_rows / max(1, len(df))

        if unique_rows < 3:
            warnings.append("Во входных данных очень мало уникальных строк. SDA может не найти устойчивую структуру.")
        elif unique_ratio < 0.05:
            warnings.append(
                f"Доля уникальных строк низкая ({unique_ratio:.2%}). "
                "Метрики качества и границы сегментов могут быть нестабильными."
            )

        return {"unique_rows": unique_rows, "unique_ratio": unique_ratio, "warnings": warnings}

    def _scale_if_needed(self, matrix: np.ndarray, do_scale: bool) -> np.ndarray:
        if not do_scale:
            return matrix
        return StandardScaler().fit_transform(matrix)

    @staticmethod
    def _check_cancel(is_cancelled):
        if is_cancelled and is_cancelled():
            raise RuntimeError("Задача отменена пользователем.")

    # =========================
    # SDA params
    # =========================

    def _extract_sda_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {
            "n_jobs", "scale", "verbose", "random_state",
            "st1_calc_quality", "n_clusters_min", "n_clusters_max", "n_clusters",
            "k_neighbours_min", "k_neighbours_max", "k_neighbours",
            "st1_merging", "st1_len_thresholds", "st1_dist_rate",
            "st2_calc_quality", "n_cl_max_thr", "k_neighb_max_thr",
            "n_edge_clusters_min", "n_edge_clusters_max", "n_edge_clusters",
            "st2_merging", "st2_len_thresholds", "st2_dist_rate",
        }
        return {k: v for k, v in params.items() if k in allowed}

    def _sanitize_sda_params(self, params: Dict[str, Any], n_rows: int) -> Dict[str, Any]:
        out = dict(params)

        max_k = max(2, n_rows - 1)
        requested_max_clusters = int(out.get("n_clusters_max", min(20, n_rows)))
        max_clusters = max(2, min(n_rows, requested_max_clusters))

        out["n_clusters_min"] = min(int(out.get("n_clusters_min", 2)), max_clusters)
        out["n_clusters_max"] = max(out["n_clusters_min"], max_clusters)

        out["k_neighbours_min"] = min(int(out.get("k_neighbours_min", 2)), max_k)
        out["k_neighbours_max"] = min(int(out.get("k_neighbours_max", max_k)), max_k)
        out["k_neighbours_max"] = max(out["k_neighbours_max"], out["k_neighbours_min"])

        if "n_edge_clusters_min" in out:
            out["n_edge_clusters_min"] = min(int(out["n_edge_clusters_min"]), max_clusters)
        if "n_edge_clusters_max" in out:
            out["n_edge_clusters_max"] = min(int(out["n_edge_clusters_max"]), max_clusters)
            out["n_edge_clusters_max"] = max(out["n_edge_clusters_max"], int(out.get("n_edge_clusters_min", 2)))

        if "n_cl_max_thr" in out:
            out["n_cl_max_thr"] = [
                max(2, min(int(v), max_clusters))
                for v in self._as_list(out["n_cl_max_thr"])
            ]
        if "k_neighb_max_thr" in out:
            out["k_neighb_max_thr"] = [
                max(2, min(int(v), max_k))
                for v in self._as_list(out["k_neighb_max_thr"])
            ]

        if "st1_len_thresholds" in out:
            out["st1_len_thresholds"] = [
                max(0, min(int(v), n_rows))
                for v in self._as_list(out["st1_len_thresholds"])
            ]
        if "st2_len_thresholds" in out:
            out["st2_len_thresholds"] = [
                max(0, min(int(v), n_rows))
                for v in self._as_list(out["st2_len_thresholds"])
            ]

        out.setdefault("n_jobs", -1)
        return out

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, range):
            return list(value)
        return [value]

    # =========================
    # Result selection
    # =========================

    def _augment_results_table(self, results_table: pd.DataFrame, n_rows: int) -> pd.DataFrame:
        out = results_table.copy()
        out["candidate_n_edges"] = out.get("St_edges", pd.Series([None] * len(out))).apply(
            lambda value: len(self._extract_edges_safe(value, n_rows))
        )
        out["candidate_n_segments"] = out["candidate_n_edges"] + 1
        return out

    def _select_best_result(
        self,
        results_table: pd.DataFrame,
        n_rows: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if results_table is None or results_table.empty:
            raise ValueError("Невозможно выбрать лучший результат: таблица SDA пуста.")

        params = params or {}
        df = self._augment_results_table(results_table, n_rows)

        prefer_min_segments = bool(params.get("prefer_min_segments", True))
        preferred_min_segments = int(params.get("preferred_min_segments", params.get("n_clusters_min", 2)))

        candidates = df
        if prefer_min_segments and "candidate_n_segments" in df.columns:
            preferred = df[df["candidate_n_segments"] >= preferred_min_segments]
            if not preferred.empty:
                candidates = preferred

        score_col = self._best_score_column(candidates)
        if score_col is not None:
            if score_col == "Avg-Dav-Bold":
                sorted_df = candidates.sort_values(
                    by=[score_col, "candidate_n_segments"],
                    ascending=[True, False],
                )
            else:
                sorted_df = candidates.sort_values(
                    by=[score_col, "candidate_n_segments"],
                    ascending=[False, False],
                )
        else:
            sorted_df = candidates.sort_values(by="candidate_n_segments", ascending=False)

        return sorted_df.iloc[0].to_dict()

    @staticmethod
    def _best_score_column(df: pd.DataFrame) -> Optional[str]:
        for col in ["Avg-Silh", "Avg-Cal-Har", "Avg-Dav-Bold"]:
            if col in df.columns and df[col].notna().any():
                return col
        return None

    @staticmethod
    def _quality_score(row: Dict[str, Any]) -> float:
        for key in ["Avg-Silh", "Avg-Cal-Har"]:
            value = row.get(key)
            try:
                if value is not None and not pd.isna(value):
                    return float(value)
            except Exception:
                continue
        return 0.0

    # =========================
    # Edges / stages / tables
    # =========================

    def _extract_edges_safe(self, st_edges_value: Any, n_rows: int) -> List[int]:
        try:
            return self._extract_edges(st_edges_value, n_rows)
        except Exception:
            return []

    def _extract_edges(self, st_edges_value: Any, n_rows: int) -> List[int]:
        if st_edges_value is None:
            return []

        if isinstance(st_edges_value, str):
            text = st_edges_value.strip()
            if not text or text.lower() in {"nan", "none"}:
                return []
            parsed = ast.literal_eval(text)
        else:
            parsed = st_edges_value

        if isinstance(parsed, (int, float, np.integer, np.floating)):
            if pd.isna(parsed):
                return []
            edges = [int(parsed)]
        else:
            edges = []
            for item in parsed:
                if pd.isna(item):
                    continue
                edges.append(int(item))

        unique_sorted = sorted(set(edges))
        return [e for e in unique_sorted if 0 < e < n_rows]

    def _edges_to_stage_ids(self, n_rows: int, edges: List[int]) -> List[int]:
        labels = np.zeros(n_rows, dtype=int)
        boundaries = [0] + list(edges) + [n_rows]
        for stage_id, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            labels[start:end] = stage_id
        return labels.tolist()

    def _build_segmented_data(
        self,
        df: pd.DataFrame,
        stage_ids: List[int],
        edges: List[int],
        timestamp_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        out = df.copy()
        out["stage_id"] = stage_ids
        out["is_boundary"] = False

        if edges:
            valid_edges = [e for e in edges if 0 <= e < len(out)]
            out.iloc[valid_edges, out.columns.get_loc("is_boundary")] = True

        out["source_index"] = out.index

        if timestamp_series is not None:
            aligned_ts = timestamp_series.reindex(out.index)
            out["timestamp"] = aligned_ts

        return out.reset_index(drop=True)

    def _build_segments_table(self, segmented_data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        grouped = segmented_data.groupby("stage_id", sort=True)

        for stage_id, group in grouped:
            row: Dict[str, Any] = {
                "segment_id": int(stage_id),
                "start_idx": int(group.index.min()),
                "end_idx": int(group.index.max()),
                "length": int(len(group)),
                "source_start_idx": int(group["source_index"].iloc[0]) if "source_index" in group.columns else int(group.index.min()),
                "source_end_idx": int(group["source_index"].iloc[-1]) if "source_index" in group.columns else int(group.index.max()),
            }

            if "timestamp" in group.columns:
                row["start_time"] = group["timestamp"].iloc[0]
                row["end_time"] = group["timestamp"].iloc[-1]

            # Для кластеризации сегментов полезнее иметь агрегаты по всем выбранным признакам,
            # а не только по первым трем.
            for col in feature_columns:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_std"] = float(group[col].std(ddof=0))
                row[f"{col}_min"] = float(group[col].min())
                row[f"{col}_max"] = float(group[col].max())

            rows.append(row)

        return pd.DataFrame(rows)

    # =========================
    # Summary / warnings
    # =========================

    def _build_summary(
        self,
        results_table: pd.DataFrame,
        best_row: Dict[str, Any],
        edges: List[int],
        segments_table: pd.DataFrame,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        summary = {
            "candidates": int(len(results_table)),
            "n_segments": int(len(segments_table)),
            "n_boundaries": int(len(edges)),
            "best_candidate_segments": int(best_row.get("candidate_n_segments", len(edges) + 1)),
            "preferred_min_segments": int(params.get("preferred_min_segments", params.get("n_clusters_min", 2))),
        }

        for key in ["Avg-Silh", "Avg-Cal-Har", "Avg-Dav-Bold", "N_stages", "Avg_stage_length"]:
            if key in best_row:
                value = best_row[key]
                if isinstance(value, (np.integer, np.floating)):
                    value = value.item()
                summary[key] = value

        return summary

    def _result_warnings(
        self,
        summary: Dict[str, Any],
        params: Dict[str, Any],
        segments_table: pd.DataFrame,
    ) -> List[str]:
        warnings: List[str] = []

        n_segments = int(summary.get("n_segments", 0))
        preferred = int(params.get("preferred_min_segments", params.get("n_clusters_min", 2)))

        if n_segments < preferred:
            warnings.append(
                f"Итоговых сегментов ({n_segments}) меньше, чем n_clusters_min ({preferred}). "
                "Это возможно из-за объединения стадий SDA или отсутствия устойчивого более детального разбиения."
            )

        silh = summary.get("Avg-Silh")
        try:
            if silh is not None and not pd.isna(silh) and float(silh) < 0.05:
                warnings.append(
                    "Avg-Silh очень низкий. Границы сегментов могут быть слабо выражены для выбранных признаков."
                )
        except Exception:
            pass

        if segments_table is not None and not segments_table.empty and "length" in segments_table.columns:
            lengths = segments_table["length"].astype(float)
            if len(lengths) > 1 and lengths.min() <= max(2, 0.01 * lengths.sum()):
                warnings.append(
                    "Есть очень короткие сегменты. Возможно, стоит увеличить минимальную длину сегмента "
                    "или усилить сглаживание/агрегацию признаков."
                )
            if len(lengths) == 1:
                warnings.append(
                    "Получился один сегмент. SDA не нашел устойчивых внутренних границ при текущих параметрах."
                )

        return warnings
