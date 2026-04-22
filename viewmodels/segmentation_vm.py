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
            source_df = self._get_source_df()
            if source_df is None:
                raise ValueError("Нет данных признаков. Сначала выполните формирование признаков.")

            timestamp_series = self._get_timestamp_series(source_df)

            result = self.segmentation_service.run_segmentation(
                features_df=source_df,
                selected_columns=selected_columns,
                params=params,
                input_kind="features",
                timestamp_series=timestamp_series,
            )

            self.current_result = result
            self.project.set_segments(result.segmented_data, params=result.params)
            self.project.parameters["segmentation_result"] = {
                "best_result": result.best_result_row,
                "edges": result.edges,
                "summary": result.summary,
                "selected_columns": result.selected_columns,
            }
            self.segmentation_ready.emit(result)
            self.info_changed.emit("Сегментация SDA завершена успешно.")
        except Exception as exc:
            self.error_occurred.emit(str(exc))

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
