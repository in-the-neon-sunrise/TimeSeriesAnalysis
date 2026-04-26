from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from PySide6.QtCore import Signal

from services.report_service import ReportGenerationOptions, ReportService
from viewmodels.base_vm import BaseViewModel


class ReportViewModel(BaseViewModel):
    report_generated = Signal(str)

    def __init__(self, project_service, report_service: ReportService | None = None):
        super().__init__()
        self.project = project_service
        self.report_service = report_service or ReportService(project_service)
        self.last_report_path: str = ""

    def get_available_stage_flags(self) -> Dict[str, bool]:
        return {
            "include_data_overview": self.project.raw_data is not None and not self.project.raw_data.empty,
            "include_primary_analysis": self.project.raw_data is not None and not self.project.raw_data.empty,
            "include_preprocessing": self.project.processed_data is not None and not self.project.processed_data.empty,
            "include_feature_extraction": self.project.features is not None and not self.project.features.empty,
            "include_segmentation": self.project.segments is not None and not self.project.segments.empty,
            "include_clustering": self.project.clusters is not None and not self.project.clusters.empty,
            "include_markov_modeling": (
                self.project.markov_matrix is not None and not self.project.markov_matrix.empty
            ) or bool(self.project.markov_result),
        }

    def generate_report(self, output_path: str, options_payload: Dict):
        try:
            if not output_path:
                raise ValueError("Не указан путь для PDF-отчёта.")

            options = ReportGenerationOptions.from_dict(options_payload)
            generated_path = self.report_service.generate_report(output_path, options)
            self.last_report_path = generated_path
            self.project.parameters.setdefault("report", {})["last_report_path"] = generated_path

            self.info_changed.emit(f"Отчёт успешно сформирован: {generated_path}")
            self.report_generated.emit(generated_path)
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def open_report_directory(self) -> str:
        if not self.last_report_path:
            raise ValueError("Отчёт ещё не сформирован.")
        return str(Path(self.last_report_path).parent)

    @staticmethod
    def normalize_output_path(path: str, default_name: str = "report.pdf") -> str:
        if not path:
            return ""
        output = Path(path)
        if output.is_dir():
            output = output / default_name
        if output.suffix.lower() != ".pdf":
            output = output.with_suffix(".pdf")
        return os.fspath(output)