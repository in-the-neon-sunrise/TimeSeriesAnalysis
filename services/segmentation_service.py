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
            raise ValueError("Нет данных признаков для сегментации.")

        selected_columns = selected_columns or list(features_df.select_dtypes(include="number").columns)
        mode = str(params.get("mode", "full")).lower()
        if mode not in {"full", "chunked"}:
            raise ValueError("Некорректный режим сегментации. Доступно: full/chunked.")
        data_for_sda, used_columns = self._prepare_numeric_data(features_df, selected_columns)
        if data_for_sda.shape[0] < 3:
            raise ValueError("Для сегментации нужно минимум 3 строки данных.")
        if self._has_only_constant_columns(data_for_sda):
            raise ValueError("Все выбранные признаки константные. Сегментация невозможна.")

        sda_params = self._extract_sda_params(params)
        sda_params = self._sanitize_sda_params(sda_params, n_rows=len(data_for_sda))
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
                chunk_size=int(params.get("chunk_size", 7200)),
                overlap=int(params.get("overlap", 600)),
                min_segment_len=int(params.get("min_segment_len", 120)),
                merge_boundaries_tolerance=int(params.get("merge_boundaries_tolerance", 120)),
                min_score_to_split=float(params.get("min_score_to_split", 0.05)),
                timestamp_series=timestamp_series,
                progress_callback=progress_callback,
                is_cancelled=is_cancelled,
            )
            results_table = result_payload["results_table"]
            stage1_results = result_payload["stage1_results"]
            edges = result_payload["edges"]
            best_row = result_payload["best_row"]
            chunk_stats = result_payload["chunk_stats"]
        else:
            results_table, stage1_results = self.adapter.run(features_for_algo, sda_params)
            if results_table is None or results_table.empty:
                raise ValueError("SDA вернул пустой результат")

            if progress_callback:
                progress_callback.emit(70, "SDA завершен, формирование результатов")
            self._check_cancel(is_cancelled)

            best_row = self._select_best_result(results_table)
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
        summary = self._build_summary(results_table, best_row, edges, segments_table)

        if chunk_stats:
            summary["chunk_stats"] = chunk_stats
            summary["mode"] = "chunked"
            summary["chunks_processed"] = len(chunk_stats)
        else:
            summary["mode"] = "full"

        if progress_callback:
            progress_callback.emit(100, "Сегментация завершена")

        return SegmentationResult(
            input_kind=input_kind,
            selected_columns=used_columns,
            params=params,
            results_table=results_table.reset_index(drop=True),
            stage1_results=stage1_results.reset_index(drop=True),
            best_result_row=best_row,
            edges=edges,
            stage_ids=stage_ids,
            segmented_data=segmented_data,
            segments_table=segments_table,
            summary=summary,
            timestamp_column=timestamp_series.name if timestamp_series is not None else None,
        )

    def run_chunked_segmentation(
            self, data, selected_columns, params, chunk_size, overlap, min_segment_len,
            merge_boundaries_tolerance, min_score_to_split, timestamp_series=None,
            progress_callback=None, is_cancelled=None,
    ) -> Dict[str, Any]:
        n_rows = len(data)
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "Некорректные параметры фрагментации: требуется chunk_size > 0 и 0 <= overlap < chunk_size.")
        step = chunk_size - overlap
        if step <= 0:
            raise ValueError("Некорректные параметры фрагментации: шаг должен быть больше 0.")

        global_edges: List[int] = [0, n_rows]
        all_results = []
        all_stage1 = []
        chunk_stats = []
        starts = list(range(0, n_rows, step))
        total_chunks = len(starts)

        for i, start in enumerate(starts, start=1):
            self._check_cancel(is_cancelled)
            end = min(start + chunk_size, n_rows)
            chunk = data.iloc[start:end]
            if len(chunk) < 3:
                chunk_stats.append({"chunk_id": i, "start": start, "end": end, "status": "skipped_too_small"})
                continue
            try:
                chunk_params = self._sanitize_sda_params(dict(params), n_rows=len(chunk))
                chunk_features = self._scale_if_needed(chunk.values, bool(params.get("scale", False)))
                chunk_params["scale"] = False
                result_tbl, st1_tbl = self.adapter.run(chunk_features, chunk_params)
                if result_tbl is None or result_tbl.empty:
                    chunk_stats.append({"chunk_id": i, "start": start, "end": end, "status": "skipped_empty_result"})
                    continue
                best_row = self._select_best_result(result_tbl)
                score = float(best_row.get("Avg-Silh", best_row.get("Avg-Cal-Har", 0.0)) or 0.0)
                local_edges = self._extract_edges(best_row.get("St_edges"), len(chunk))
                if score >= min_score_to_split:
                    global_edges.extend([start + edge for edge in local_edges if 0 < edge < len(chunk)])
                    status = "accepted"
                else:
                    status = "low_score_skipped"
                chunk_stats.append({"chunk_id": i, "start": start, "end": end, "score": score, "status": status})
                result_tbl = result_tbl.copy()
                result_tbl["chunk_id"] = i
                all_results.append(result_tbl)
                if st1_tbl is not None and not st1_tbl.empty:
                    st1_tbl = st1_tbl.copy()
                    st1_tbl["chunk_id"] = i
                    all_stage1.append(st1_tbl)
            except Exception:
                chunk_stats.append({"chunk_id": i, "start": start, "end": end, "status": "failed"})
            if progress_callback:
                progress = int(15 + (i / total_chunks) * 75)
                progress_callback.emit(progress, f"Обработка фрагмента {i} из {total_chunks}")

        if not all_results:
            raise ValueError("SDA не смог обработать ни одного фрагмента.")
        edges = self.normalize_edges(global_edges, n_rows)
        edges = self.merge_close_edges(edges, merge_boundaries_tolerance, n_rows)
        edges = self.remove_short_segments(edges, min_segment_len, n_rows)

        results_table = pd.concat(all_results, ignore_index=True)
        stage1_results = pd.concat(all_stage1, ignore_index=True) if all_stage1 else pd.DataFrame()
        best_row = self._select_best_result(results_table)
        if progress_callback:
            progress_callback.emit(90,
                                   f"Сегментация завершена: обработано {total_chunks} фрагментов, найдено {len(edges) - 1} сегмента")
        return {"results_table": results_table, "stage1_results": stage1_results, "edges": edges[1:-1],
                "best_row": best_row, "chunk_stats": chunk_stats}

    def normalize_edges(self, edges: List[int], n_rows: int) -> List[int]:
        out = [int(e) for e in edges if 0 <= int(e) <= n_rows]
        out.extend([0, n_rows])
        return sorted(set(out))

    def merge_close_edges(self, edges: List[int], tolerance: int, n_rows: int) -> List[int]:
        if tolerance <= 0:
            return self.normalize_edges(edges, n_rows)
        merged = [edges[0]]
        for edge in edges[1:]:
            if edge - merged[-1] <= tolerance and edge not in (0, n_rows):
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

    @staticmethod
    def _has_only_constant_columns(df: pd.DataFrame) -> bool:
        return all(df[c].nunique(dropna=True) <= 1 for c in df.columns)

    @staticmethod
    def _check_cancel(is_cancelled):
        if is_cancelled and is_cancelled():
            raise RuntimeError("Задача отменена пользователем.")

    def _prepare_numeric_data(
        self,
        df: pd.DataFrame,
        selected_columns: Iterable[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        missing = [c for c in selected_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Выбранные столбцы отсутствуют в данных: {missing}")

        subset = df.loc[:, list(selected_columns)]
        numeric = subset.select_dtypes(include="number")
        if numeric.empty:
            raise ValueError("После фильтрации не осталось числовых признаков.")

        cleaned = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if cleaned.empty:
            raise ValueError("Все строки содержат NaN/Inf после очистки признаков.")

        return cleaned, list(cleaned.columns)

    def _scale_if_needed(self, matrix: np.ndarray, do_scale: bool) -> np.ndarray:
        if not do_scale:
            return matrix
        return StandardScaler().fit_transform(matrix)

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

    def _select_best_result(self, results_table: pd.DataFrame) -> Dict[str, Any]:
        if "Avg-Silh" in results_table.columns:
            sorted_df = results_table.sort_values(by="Avg-Silh", ascending=False)
        elif "Avg-Cal-Har" in results_table.columns:
            sorted_df = results_table.sort_values(by="Avg-Cal-Har", ascending=False)
        else:
            sorted_df = results_table
        return sorted_df.iloc[0].to_dict()

    def _sanitize_sda_params(self, params: Dict[str, Any], n_rows: int) -> Dict[str, Any]:
        out = dict(params)
        max_k = max(2, n_rows - 1)
        max_clusters = max(2, min(n_rows, out.get("n_clusters_max", n_rows)))

        out["n_clusters_min"] = min(out.get("n_clusters_min", 2), max_clusters)
        out["n_clusters_max"] = max(out["n_clusters_min"], max_clusters)

        out["k_neighbours_min"] = min(out.get("k_neighbours_min", 2), max_k)
        out["k_neighbours_max"] = min(out.get("k_neighbours_max", max_k), max_k)
        out["k_neighbours_max"] = max(out["k_neighbours_max"], out["k_neighbours_min"])

        if "n_edge_clusters_min" in out:
            out["n_edge_clusters_min"] = min(out["n_edge_clusters_min"], max_clusters)
        if "n_edge_clusters_max" in out:
            out["n_edge_clusters_max"] = min(out["n_edge_clusters_max"], max_clusters)
            out["n_edge_clusters_max"] = max(out["n_edge_clusters_max"], out.get("n_edge_clusters_min", 2))
        return out

    def _extract_edges(self, st_edges_value: Any, n_rows: int) -> List[int]:
        if st_edges_value is None:
            return []

        if isinstance(st_edges_value, str):
            parsed = ast.literal_eval(st_edges_value)
        else:
            parsed = st_edges_value

        if isinstance(parsed, (int, float, np.integer, np.floating)):
            edges = [int(parsed)]
        else:
            edges = [int(x) for x in parsed]

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
            }

            if "timestamp" in group.columns:
                row["start_time"] = group["timestamp"].iloc[0]
                row["end_time"] = group["timestamp"].iloc[-1]

            for col in feature_columns[:3]:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_std"] = float(group[col].std(ddof=0))

            rows.append(row)

        return pd.DataFrame(rows)

    def _build_summary(
        self,
        results_table: pd.DataFrame,
        best_row: Dict[str, Any],
        edges: List[int],
        segments_table: pd.DataFrame,
    ) -> Dict[str, Any]:
        summary = {
            "candidates": len(results_table),
            "n_segments": int(len(segments_table)),
            "n_boundaries": int(len(edges)),
        }
        for key in ["Avg-Silh", "Avg-Cal-Har", "Avg-Dav-Bold", "N_stages", "Avg_stage_length"]:
            if key in best_row:
                summary[key] = best_row[key]
        return summary
