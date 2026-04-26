from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from PySide6.QtCore import Signal

from core.segmentation.segmentation_models import SegmentationResult
from services.segmentation_service import SegmentationService
from viewmodels.base_vm import BaseViewModel


class SegmentationViewModel(BaseViewModel):
    segmentation_ready = Signal(object)
    columns_ready = Signal(list)

    def __init__(self, project_service, segmentation_service: Optional[SegmentationService] = None):
        super().__init__()
        self.project = project_service
        self.segmentation_service = segmentation_service or SegmentationService()
        self.current_result: Optional[SegmentationResult] = None

    def load_available_columns(self):
        df = self._get_source_df()
        if df is None:
            self.columns_ready.emit([])
            return
        columns = list(df.select_dtypes(include="number").columns)
        self.columns_ready.emit(columns)

    def run_segmentation(self, selected_columns: List[str], params: Dict[str, Any]):
        try:
            request = self.build_segmentation_request(selected_columns, params)
            result = self.execute_segmentation(**request)
            self.apply_segmentation_result(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def build_segmentation_request(self, selected_columns: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
        source_df = self._get_source_df()
        if source_df is None:
            raise ValueError("Нет данных признаков. Сначала выполните формирование признаков")

        return {
            "source_df": source_df,
            "selected_columns": selected_columns,
            "params": params,
            "timestamp_series": self._get_timestamp_series(source_df),
        }

    def execute_segmentation(self, source_df, selected_columns, params, timestamp_series, progress_callback=None,
                             is_cancelled=None):
        return self.segmentation_service.run_segmentation(
            features_df=source_df,
            selected_columns=selected_columns,
            params=params,
            input_kind="features",
            timestamp_series=timestamp_series,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )

    def apply_segmentation_result(self, result: SegmentationResult):
        self.current_result = result
        self.project.set_segments(result.segmented_data, params=result.params)
        self.project.parameters["segmentation_result"] = {
            "best_result": result.best_result_row,
            "edges": result.edges,
            "summary": result.summary,
            "selected_columns": result.selected_columns,
        }
        self.segmentation_ready.emit(result)
        self.info_changed.emit("Сегментация SDA завершена успешно")

    def reset_result(self):
        self.current_result = None
        self.project.set_segments(None, params={})
        self.project.parameters.pop("segmentation", None)
        self.project.parameters.pop("segmentation_result", None)
        self.info_changed.emit("Результат сегментации очищен.")

    def _get_source_df(self) -> Optional[pd.DataFrame]:
        if self.project.features is not None and not self.project.features.empty:
            return self.project.features
        if self.project.processed_data is not None and not self.project.processed_data.empty:
            return self.project.processed_data
        return None

    def _get_timestamp_series(self, source_df: pd.DataFrame) -> Optional[pd.Series]:
        raw = self.project.raw_data
        if raw is None or raw.empty:
            return None

        candidates = [c for c in raw.columns if "time" in c.lower() or "date" in c.lower()]
        if not candidates:
            datetime_cols = [c for c in raw.columns if str(raw[c].dtype).startswith("datetime")]
            candidates = datetime_cols
        if not candidates:
            return None

        col = candidates[0]
        if len(raw[col]) == len(source_df):
            return raw[col]
        return None
