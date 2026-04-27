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

        numeric_columns = list(segments_df.select_dtypes(include="number").columns)
        self.source_info_ready.emit(
            {
                "available": True,
                "message": "Данные сегментации готовы для clustering.",
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

    def build_clustering_request(self, method: str, selected_columns: List[str], params: Dict[str, Any]) -> Dict[
        str, Any]:
        segments_df = self.project.segments
        if segments_df is None or segments_df.empty:
            raise ValueError("Нет данных сегментации. Сначала выполните этап segmentation.")
        return {
            "segments_df": segments_df,
            "method": method,
            "selected_columns": selected_columns,
            "params": params,
        }

    def execute_clustering(self, segments_df, method, selected_columns, params, progress_callback=None,
                           is_cancelled=None):
        return self.clustering_service.run_clustering(
            segments_df=segments_df,
            method=method,
            selected_columns=selected_columns,
            params=params,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )

    def apply_clustering_result(self, result: ClusteringResult):
        self.current_result = result
        self._persist_result(result)
        self.clustering_ready.emit(result)
        self.info_changed.emit("Кластеризация сегментов успешно выполнена.")

    def reset_result(self):
        self.current_result = None
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
        }
