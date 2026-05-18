from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from PySide6.QtCore import Signal

from core.clustering.clustering_models import ClusteringResult
from services.clustering_service import ClusteringService
from viewmodels.base_vm import BaseViewModel


class ClusteringViewModel(BaseViewModel):
    source_info_ready = Signal(dict)
    columns_ready = Signal(list)
    clustering_ready = Signal(object)
    result_reset = Signal()

    def __init__(self, project_service, clustering_service: Optional[ClusteringService] = None):
        super().__init__()
        self.project = project_service
        self.clustering_service = clustering_service or ClusteringService()
        self.current_result: Optional[ClusteringResult] = None
        self.last_k_evaluation: Optional[pd.DataFrame] = None

    def refresh(self):
        segments_df = self.project.segments
        if segments_df is None or segments_df.empty:
            self.source_info_ready.emit(
                {
                    "available": False,
                    "message": "Нет результатов сегментации. Сначала выполните segmentation.",
                    "segments_count": 0,
                }
            )
            self.columns_ready.emit([])
            return

        numeric_columns = self._default_feature_columns(segments_df)
        self.source_info_ready.emit(
            {
                "available": True,
                "message": "Данные сегментации готовы для clustering. Каждая строка — один сегмент.",
                "segments_count": int(len(segments_df)),
            }
        )
        self.columns_ready.emit(numeric_columns)

    def run_clustering(self, method: str, selected_columns: List[str], params: Dict[str, Any]):
        try:
            request = self.build_clustering_request(method, selected_columns, params)
            result = self.execute_clustering(**request)
            self.apply_clustering_result(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def build_clustering_request(
        self,
        method: str,
        selected_columns: List[str],
        params: Dict[str, Any],
        source_key: str | None = None,
        output_name: str | None = None,
    ) -> Dict[str, Any]:
        segments_df = self._get_source_df(source_key)
        if segments_df is None or segments_df.empty:
            raise ValueError("Нет подходящих данных для кластеризации.")
        return {
            "segments_df": segments_df,
            "method": method,
            "selected_columns": selected_columns,
            "params": params,
            "source_key": source_key,
            "output_name": output_name,
        }

    def execute_clustering(
        self,
        segments_df,
        method,
        selected_columns,
        params,
        source_key=None,
        output_name=None,
        progress_callback=None,
        is_cancelled=None,
    ):
        result = self.clustering_service.run_clustering(
            segments_df=segments_df,
            method=method,
            selected_columns=selected_columns,
            params=params,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )
        result.params = dict(result.params)
        result.params["source_dataset_name"] = source_key or "segments"
        result.params["output_dataset_name"] = output_name or "clusters"
        return result

    def evaluate_kmeans_range(
        self,
        selected_columns: List[str],
        params: Dict[str, Any],
        k_min: int,
        k_max: int,
    ) -> pd.DataFrame:
        segments_df = self._get_source_df("segments")
        evaluation = self.clustering_service.evaluate_kmeans_range(
            segments_df=segments_df,
            selected_columns=selected_columns,
            params=params,
            k_min=k_min,
            k_max=k_max,
        )
        self.last_k_evaluation = evaluation
        return evaluation

    def apply_clustering_result(self, result: ClusteringResult):
        self.current_result = result
        self._persist_result(result)
        self.clustering_ready.emit(result)
        self.info_changed.emit("Кластеризация сегментов успешно выполнена.")

    def reset_result(self):
        self.current_result = None
        self.last_k_evaluation = None
        self.project.set_clusters(None, params={})
        self.project.clustering_result = None
        self.project.parameters.pop("clustering", None)
        self.result_reset.emit()
        self.info_changed.emit("Результаты кластеризации очищены.")

    def export_clustered_segments(self, file_path: str):
        if self.current_result is None:
            raise ValueError("Нет результатов для экспорта.")
        if not file_path:
            return
        self.current_result.clustered_segments.to_csv(Path(file_path), index=False)

    def _persist_result(self, result: ClusteringResult):
        self.project.set_clusters(result.clustered_segments, params=result.params)
        self.project.clustering_result = result.to_project_payload()
        self.project.parameters["clustering"] = {
            "method": result.method,
            "params": dict(result.params),
            "selected_columns": list(result.selected_columns),
            "metrics": dict(result.metrics),
            "summary": dict(result.summary),
            "distance_metric": result.distance_metric,
            "source_info": dict(result.source_info),
            "k_evaluation": self.last_k_evaluation.to_dict(orient="records")
            if self.last_k_evaluation is not None
            else [],
        }

    def _get_source_df(self, source_key: str | None = None):
        if source_key and source_key != "segments":
            raise ValueError("Кластеризация поддерживает только входные данные сегментов. Сначала выполните сегментацию.")
        return self.project.segments

    def _default_feature_columns(self, segments_df):
        return self.clustering_service.default_feature_columns(segments_df)
