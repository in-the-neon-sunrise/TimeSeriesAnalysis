from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal

from core.markov.markov_models import MarkovResult
from services.markov_service import MarkovService
from viewmodels.base_vm import BaseViewModel


class MarkovViewModel(BaseViewModel):
    source_info_ready = Signal(dict)
    model_ready = Signal(object)
    model_reset = Signal()

    def __init__(self, project_service, markov_service: Optional[MarkovService] = None):
        super().__init__()
        self.project = project_service
        self.markov_service = markov_service or MarkovService()
        self.current_result: Optional[MarkovResult] = None

    def refresh_source_info(self):
        try:
            source_df = self._get_source_df()
            if source_df is None or source_df.empty:
                self.source_info_ready.emit(
                    {
                        "available": False,
                        "message": "Нет результатов кластеризации. Сначала выполните этап clustering.",
                        "sequence_length": 0,
                        "unique_states": 0,
                        "source_name": "clusters",
                    }
                )
                return

            sequence = self.markov_service.extract_state_sequence(source_df)
            self.source_info_ready.emit(
                {
                    "available": True,
                    "message": "Данные кластеризации готовы для построения цепи Маркова.",
                    "sequence_length": len(sequence),
                    "unique_states": len(set(sequence)),
                    "source_name": "clusters",
                }
            )
        except Exception as exc:
            self.source_info_ready.emit(
                {
                    "available": False,
                    "message": str(exc),
                    "sequence_length": 0,
                    "unique_states": 0,
                    "source_name": "clusters",
                }
            )

    def build_model(self, order: int, normalize: bool, sequential_only: bool, min_frequency: int):
        try:
            request = self.build_model_request(order, normalize, sequential_only, min_frequency)
            result = self.execute_model(**request)
            self.apply_model_result(result)
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def build_model_request(self, order: int, normalize: bool, sequential_only: bool, min_frequency: int):
        source_df = self._get_source_df()
        if source_df is None or source_df.empty:
            raise ValueError("Нет данных кластеризации. Запустите clustering перед Markov modeling.")
        return {
            "source_df": source_df,
            "order": order,
            "normalize": normalize,
            "sequential_only": sequential_only,
            "min_frequency": min_frequency,
        }

    def execute_model(self, source_df, order, normalize, sequential_only, min_frequency, progress_callback=None,
                      is_cancelled=None):
        return self.markov_service.build_model(
            source_df=source_df,
            order=order,
            normalize=normalize,
            sequential_only=sequential_only,
            min_frequency=min_frequency,
            progress_callback=progress_callback,
            is_cancelled=is_cancelled,
        )

    def apply_model_result(self, result: MarkovResult):
        self.current_result = result
        self._persist_result(result)
        self.model_ready.emit(result)
        self.info_changed.emit("Markov model успешно построена.")

    def reset_result(self):
        self.current_result = None
        self.project.markov_matrix = None
        self.project.markov_result = None
        self.project.parameters.pop("markov", None)
        self.model_reset.emit()
        self.info_changed.emit("Результаты Markov modeling очищены.")

    def export_probabilities_csv(self, file_path: str):
        if self.current_result is None:
            raise ValueError("Нет результатов для экспорта.")
        if not file_path:
            return

        output_path = Path(file_path)
        self.current_result.transition_probabilities.to_csv(output_path)

    def _get_source_df(self):
        return self.project.clusters

    def _persist_result(self, result: MarkovResult):
        payload = result.to_project_payload()
        self.project.set_markov_matrix(result.transition_probabilities, params=result.params)
        self.project.set_markov_result(payload, params=result.params)
        self.project.parameters["markov"] = {
            "order": result.order,
            "params": result.params,
            "summary": result.summary,
            "stationary_distribution": result.stationary_distribution,
        }
