from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PySide6.QtCore import Signal

from services.report_service import ReportGenerationOptions, ReportService
from viewmodels.base_vm import BaseViewModel


class ReportViewModel(BaseViewModel):
    report_generated = Signal(str)
    csv_exported = Signal(str)

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

            payload = dict(options_payload)

            payload["include_plots"] = True
            payload["include_tables"] = True
            payload["include_summary"] = True

            options = ReportGenerationOptions.from_dict(payload)
            generated_path = self.report_service.generate_report(output_path, options)
            self.last_report_path = generated_path
            self.project.parameters.setdefault("report", {})["last_report_path"] = generated_path

            self.info_changed.emit(f"Отчёт сформирован: {generated_path}")
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

    def get_csv_export_items(self) -> List[Dict]:
        items: List[Dict] = []

        def add_item(key: str, title: str, filename: str, df):
            if isinstance(df, pd.DataFrame) and not df.empty:
                items.append(
                    {
                        "key": key,
                        "title": title,
                        "filename": filename,
                        "rows": int(df.shape[0]),
                        "columns": int(df.shape[1]),
                    }
                )

        add_item("raw_data", "Исходные данные", "raw_data.csv", self.project.raw_data)
        add_item("processed_data", "Предобработанные данные", "processed_data.csv", self.project.processed_data)
        add_item("features", "Матрица признаков", "features.csv", self.project.features)
        add_item("segments", "Сегменты", "segments.csv", self.project.segments)
        add_item("clusters", "Кластеризованные сегменты", "clustered_segments.csv", self.project.clusters)
        add_item("markov_matrix", "Матрица вероятностей Маркова", "markov_transition_probabilities.csv", self.project.markov_matrix)

        markov_result = self.project.markov_result
        if isinstance(markov_result, dict):
            add_item(
                "markov_counts",
                "Матрица частот переходов",
                "markov_transition_counts.csv",
                markov_result.get("transition_counts"),
            )
            add_item(
                "markov_long",
                "Таблица переходов Маркова",
                "markov_transitions_long.csv",
                markov_result.get("transitions_long_table"),
            )

            state_sequence = markov_result.get("state_sequence")
            if state_sequence:
                seq_df = pd.DataFrame(
                    {
                        "position": list(range(len(state_sequence))),
                        "state": state_sequence,
                    }
                )
                items.append(
                    {
                        "key": "markov_sequence",
                        "title": "Последовательность состояний",
                        "filename": "markov_state_sequence.csv",
                        "rows": int(seq_df.shape[0]),
                        "columns": int(seq_df.shape[1]),
                    }
                )

        return items

    def get_csv_dataframe(self, key: str):
        mapping = {
            "raw_data": self.project.raw_data,
            "processed_data": self.project.processed_data,
            "features": self.project.features,
            "segments": self.project.segments,
            "clusters": self.project.clusters,
            "markov_matrix": self.project.markov_matrix,
        }

        if key in mapping:
            return mapping[key]

        markov_result = self.project.markov_result
        if isinstance(markov_result, dict):
            if key == "markov_counts":
                return markov_result.get("transition_counts")
            if key == "markov_long":
                return markov_result.get("transitions_long_table")
            if key == "markov_sequence":
                sequence = markov_result.get("state_sequence", [])
                if sequence:
                    return pd.DataFrame(
                        {
                            "position": list(range(len(sequence))),
                            "state": sequence,
                        }
                    )

        return None

    def export_csv_items(self, selected_keys: List[str], output_dir: str) -> List[str]:
        if not selected_keys:
            raise ValueError("Выберите хотя бы один набор данных для CSV-экспорта.")
        if not output_dir:
            raise ValueError("Не выбрана папка для сохранения CSV.")

        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)

        item_meta = {item["key"]: item for item in self.get_csv_export_items()}
        exported_paths: List[str] = []

        for key in selected_keys:
            df = self.get_csv_dataframe(key)
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            filename = item_meta.get(key, {}).get("filename", f"{key}.csv")
            path = folder / filename
            df.to_csv(path, index=False)
            exported_paths.append(str(path))

        if not exported_paths:
            raise ValueError("Нет доступных данных для выбранных CSV-файлов.")

        self.csv_exported.emit(str(folder))
        self.info_changed.emit(f"CSV экспортированы: {len(exported_paths)} файл(ов)")
        return exported_paths
