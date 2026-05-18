from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ui.pages.base_page import BasePage


class ReportPage(BasePage):
    def __init__(self, report_vm):
        super().__init__()
        self.vm = report_vm
        self.stage_checkboxes: dict[str, QCheckBox] = {}
        self.csv_checkboxes: dict[str, QCheckBox] = {}
        self._last_csv_folder = ""
        self._build_ui()
        self._bind_signals()

    def _build_ui(self):
        root = QVBoxLayout(self)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setSpacing(12)

        title = QLabel("Отчёт и экспорт")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        layout.addWidget(self._build_pdf_group())
        layout.addWidget(self._build_csv_group())

        self.last_result_label = QLabel("")
        self.last_result_label.setWordWrap(True)
        layout.addWidget(self.last_result_label)

        layout.addStretch()
        root.addWidget(scroll)

    def _build_pdf_group(self):
        group = QGroupBox("PDF-отчёт")
        layout = QVBoxLayout(group)

        settings = QFormLayout()
        self.report_title_edit = QLineEdit("Отчёт по анализу временного ряда")
        self.author_edit = QLineEdit()

        settings.addRow("Название отчёта:", self.report_title_edit)
        settings.addRow("Автор/проект:", self.author_edit)
        layout.addLayout(settings)

        steps_group = QGroupBox("Этапы отчёта")
        steps_layout = QVBoxLayout(steps_group)

        self.stage_checkboxes = {
            "include_data_overview": QCheckBox("Данные"),
            "include_primary_analysis": QCheckBox("Первичный анализ"),
            "include_preprocessing": QCheckBox("Предобработка"),
            "include_feature_extraction": QCheckBox("Извлечение признаков"),
            "include_segmentation": QCheckBox("Сегментация"),
            "include_clustering": QCheckBox("Кластеризация"),
            "include_markov_modeling": QCheckBox("Цепи Маркова"),
        }

        for cb in self.stage_checkboxes.values():
            cb.setChecked(True)
            steps_layout.addWidget(cb)

        layout.addWidget(steps_group)

        actions = QHBoxLayout()
        self.generate_btn = QPushButton("Сохранить PDF-отчёт...")
        self.open_folder_btn = QPushButton("Открыть папку отчёта")
        self.open_folder_btn.setEnabled(False)

        actions.addWidget(self.generate_btn)
        actions.addWidget(self.open_folder_btn)
        layout.addLayout(actions)

        return group

    def _build_csv_group(self):
        group = QGroupBox("CSV-выгрузки")
        layout = QVBoxLayout(group)

        controls = QHBoxLayout()
        self.csv_select_all_btn = QPushButton("Выбрать все")
        self.csv_clear_all_btn = QPushButton("Снять все")
        self.export_csv_btn = QPushButton("Сохранить выбранные CSV...")

        self.csv_select_all_btn.clicked.connect(lambda: self._set_all_csv_checked(True))
        self.csv_clear_all_btn.clicked.connect(lambda: self._set_all_csv_checked(False))
        self.export_csv_btn.clicked.connect(self.on_export_csv_clicked)

        controls.addWidget(self.csv_select_all_btn)
        controls.addWidget(self.csv_clear_all_btn)
        controls.addWidget(self.export_csv_btn)
        layout.addLayout(controls)

        self.csv_list_widget = QWidget()
        self.csv_list_layout = QVBoxLayout(self.csv_list_widget)
        self.csv_list_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.csv_list_widget)

        return group

    def _bind_signals(self):
        self.generate_btn.clicked.connect(self.on_generate_clicked)
        self.open_folder_btn.clicked.connect(self.on_open_folder_clicked)

        self.vm.report_generated.connect(self.on_report_generated)
        self.vm.csv_exported.connect(self.on_csv_exported)
        self.vm.info_changed.connect(self.on_info)
        self.vm.error_occurred.connect(self.on_error)

    def on_enter(self):
        self._set_defaults_for_available_stages()
        self._render_csv_export_items()

    def _set_defaults_for_available_stages(self):
        flags = self.vm.get_available_stage_flags()

        for key, cb in self.stage_checkboxes.items():
            available = bool(flags.get(key, False))
            cb.setChecked(available)
            cb.setEnabled(available)

    def on_generate_clicked(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить PDF-отчёт",
            "report.pdf",
            "PDF (*.pdf)",
        )
        if not file_path:
            return

        output_path = self.vm.normalize_output_path(file_path)

        payload = {
            "title": self.report_title_edit.text().strip() or "Отчёт по анализу временного ряда",
            "author": self.author_edit.text().strip(),
        }

        for key, checkbox in self.stage_checkboxes.items():
            payload[key] = checkbox.isChecked()

        self.generate_btn.setEnabled(False)
        self.vm.generate_report(output_path=output_path, options_payload=payload)

    def on_report_generated(self, path: str):
        self.generate_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(True)
        self.last_result_label.setText(f"PDF сохранён: {path}")

    def on_open_folder_clicked(self):
        try:
            folder = self.vm.open_report_directory()
        except Exception as exc:
            QMessageBox.warning(self, "Открытие папки", str(exc))
            return

        self._open_folder(folder)

    def _render_csv_export_items(self):
        for cb in self.csv_checkboxes.values():
            self.csv_list_layout.removeWidget(cb)
            cb.deleteLater()
        self.csv_checkboxes.clear()

        items = self.vm.get_csv_export_items()

        if not items:
            empty = QLabel("Нет данных для CSV-экспорта.")
            empty.setObjectName("empty_csv_label")
            self.csv_list_layout.addWidget(empty)
            self.export_csv_btn.setEnabled(False)
            self.csv_select_all_btn.setEnabled(False)
            self.csv_clear_all_btn.setEnabled(False)
            return

        self.export_csv_btn.setEnabled(True)
        self.csv_select_all_btn.setEnabled(True)
        self.csv_clear_all_btn.setEnabled(True)

        for item in items:
            title = item["title"]
            rows = item["rows"]
            columns = item["columns"]
            filename = item["filename"]

            cb = QCheckBox(f"{title} — {rows}×{columns} → {filename}")
            cb.setChecked(True)
            cb.setToolTip(filename)

            self.csv_checkboxes[item["key"]] = cb
            self.csv_list_layout.addWidget(cb)

    def _set_all_csv_checked(self, checked: bool):
        for cb in self.csv_checkboxes.values():
            cb.setChecked(checked)

    def _selected_csv_keys(self):
        return [key for key, cb in self.csv_checkboxes.items() if cb.isChecked()]

    def on_export_csv_clicked(self):
        selected = self._selected_csv_keys()
        if not selected:
            QMessageBox.warning(self, "CSV-экспорт", "Выберите хотя бы один набор данных.")
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку для CSV-файлов",
            str(Path.home()),
        )
        if not folder:
            return

        try:
            paths = self.vm.export_csv_items(selected, folder)
        except Exception as exc:
            QMessageBox.critical(self, "CSV-экспорт", str(exc))
            return

        self._last_csv_folder = folder
        self.last_result_label.setText(f"CSV сохранены в папку: {folder} ({len(paths)} файл(ов))")

    def on_csv_exported(self, folder: str):
        self._last_csv_folder = folder


    def on_info(self, message: str):
        # Статусный блок убран; короткие сообщения показываем внизу страницы.
        if message:
            self.last_result_label.setText(message)

    def on_error(self, message: str):
        self.generate_btn.setEnabled(True)
        QMessageBox.critical(self, "Ошибка", message)

    @staticmethod
    def _open_folder(folder: str):
        if os.name == "nt":
            os.startfile(folder)  # type: ignore[attr-defined]
        elif os.name == "posix":
            subprocess.Popen(["xdg-open", folder])
