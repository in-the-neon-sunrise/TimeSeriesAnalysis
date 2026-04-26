import os
import subprocess

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
    QVBoxLayout,
)

from ui.pages.base_page import BasePage


class ReportPage(BasePage):
    def __init__(self, report_vm):
        super().__init__()
        self.vm = report_vm
        self._build_ui()
        self._bind_signals()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Генерация отчёта")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        content_group = QGroupBox("Содержимое отчёта")
        content_layout = QVBoxLayout(content_group)

        self.checkboxes = {
            "include_data_overview": QCheckBox("Include data overview"),
            "include_primary_analysis": QCheckBox("Include primary analysis"),
            "include_preprocessing": QCheckBox("Include preprocessing"),
            "include_feature_extraction": QCheckBox("Include feature extraction"),
            "include_segmentation": QCheckBox("Include segmentation"),
            "include_clustering": QCheckBox("Include clustering"),
            "include_markov_modeling": QCheckBox("Include Markov modeling"),
            "include_plots": QCheckBox("Include plots"),
            "include_tables": QCheckBox("Include tables"),
            "include_summary": QCheckBox("Include summary"),
        }

        for cb in self.checkboxes.values():
            cb.setChecked(True)
            content_layout.addWidget(cb)

        layout.addWidget(content_group)

        settings_group = QGroupBox("Настройки отчёта")
        settings_layout = QFormLayout(settings_group)

        self.report_title_edit = QLineEdit("Отчёт по анализу временного ряда")
        self.author_edit = QLineEdit()

        path_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Выберите путь сохранения PDF")
        self.browse_btn = QPushButton("Выбрать...")
        self.browse_btn.clicked.connect(self.on_choose_path)
        path_layout.addWidget(self.output_path_edit)
        path_layout.addWidget(self.browse_btn)

        settings_layout.addRow("Название отчёта:", self.report_title_edit)
        settings_layout.addRow("Автор/проект:", self.author_edit)
        settings_layout.addRow("Путь сохранения:", path_layout)

        layout.addWidget(settings_group)

        actions = QHBoxLayout()
        self.generate_btn = QPushButton("Сформировать PDF")
        self.open_folder_btn = QPushButton("Открыть папку с отчётом")
        self.reset_btn = QPushButton("Сбросить настройки")

        actions.addWidget(self.generate_btn)
        actions.addWidget(self.open_folder_btn)
        actions.addWidget(self.reset_btn)
        layout.addLayout(actions)

        self.status_label = QLabel("Статус: готово")
        self.path_label = QLabel("")
        self.path_label.setWordWrap(True)

        layout.addWidget(self.status_label)
        layout.addWidget(self.path_label)
        layout.addStretch()

    def _bind_signals(self):
        self.generate_btn.clicked.connect(self.on_generate_clicked)
        self.open_folder_btn.clicked.connect(self.on_open_folder_clicked)
        self.reset_btn.clicked.connect(self.on_reset_clicked)

        self.vm.report_generated.connect(self.on_report_generated)
        self.vm.info_changed.connect(self.on_info)
        self.vm.error_occurred.connect(self.on_error)

    def on_enter(self):
        self._set_defaults_for_available_stages()

    def _set_defaults_for_available_stages(self):
        flags = self.vm.get_available_stage_flags()
        for key, available in flags.items():
            cb = self.checkboxes.get(key)
            if cb is None:
                continue
            cb.setChecked(bool(available))
        self.checkboxes["include_plots"].setChecked(True)
        self.checkboxes["include_tables"].setChecked(True)
        self.checkboxes["include_summary"].setChecked(True)

    def on_choose_path(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить отчёт", "report.pdf", "PDF (*.pdf)")
        if file_path:
            self.output_path_edit.setText(self.vm.normalize_output_path(file_path))

    def on_generate_clicked(self):
        output_path = self.vm.normalize_output_path(self.output_path_edit.text().strip())
        payload = {
            "title": self.report_title_edit.text().strip() or "Отчёт по анализу временного ряда",
            "author": self.author_edit.text().strip(),
        }
        for key, checkbox in self.checkboxes.items():
            payload[key] = checkbox.isChecked()

        self.status_label.setText("Статус: формирование отчёта...")
        self.vm.generate_report(output_path=output_path, options_payload=payload)

    def on_report_generated(self, path: str):
        self.status_label.setText("Статус: отчёт сформирован успешно")
        self.path_label.setText(f"Файл: {path}")

    def on_info(self, message: str):
        self.status_label.setText(f"Статус: {message}")

    def on_error(self, message: str):
        self.status_label.setText("Статус: ошибка")
        QMessageBox.critical(self, "Ошибка генерации отчёта", message)

    def on_open_folder_clicked(self):
        try:
            folder = self.vm.open_report_directory()
        except Exception as exc:
            QMessageBox.warning(self, "Открытие папки", str(exc))
            return

        if os.name == "nt":
            os.startfile(folder)  # type: ignore[attr-defined]
        elif os.name == "posix":
            subprocess.Popen(["xdg-open", folder])

    def on_reset_clicked(self):
        self.report_title_edit.setText("Отчёт по анализу временного ряда")
        self.author_edit.clear()
        self.output_path_edit.clear()
        self._set_defaults_for_available_stages()
        self.status_label.setText("Статус: настройки сброшены")
        self.path_label.clear()