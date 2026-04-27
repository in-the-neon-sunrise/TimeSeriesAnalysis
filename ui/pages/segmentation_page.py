from __future__ import annotations

from pathlib import Path
from typing import List

from matplotlib.figure import Figure
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QFileDialog,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
    QProgressBar
)

from ui.canvas import ScrollFriendlyCanvas
from ui.models.dataframe_model import DataFrameModel
from ui.pages.base_page import BasePage
from viewmodels.segmentation_vm import SegmentationViewModel
from workers.pipeline_worker import PipelineWorker


class SegmentationPage(BasePage):
    def __init__(self, data_vm):
        super().__init__()
        self.data_vm = data_vm
        self.vm = SegmentationViewModel(data_vm.project)

        self.vm.columns_ready.connect(self._render_columns)
        self.vm.segmentation_ready.connect(self._show_result)
        self.vm.error_occurred.connect(self._show_error)
        self.vm.info_changed.connect(self._show_info)

        self.column_checkboxes: List[QCheckBox] = []
        self.current_result = None

        self.thread_pool = QThreadPool.globalInstance()
        self.current_worker = None

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setSpacing(12)

        title = QLabel("Сегментация (SDA)")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_params_group())
        layout.addWidget(self._build_advanced_group())
        layout.addLayout(self._build_actions())

        self.status_label = QLabel("Статус: ожидание")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self._build_results_group())
        layout.addStretch()

        root = QVBoxLayout(self)
        root.addWidget(scroll)

    def on_enter(self):
        self.vm.load_available_columns()

    def _build_input_group(self):
        group = QGroupBox("Входные данные")
        vbox = QVBoxLayout(group)

        self.input_source_label = QLabel("Источник: features")
        vbox.addWidget(self.input_source_label)

        btn_row = QHBoxLayout()
        self.select_all_btn = QPushButton("Выбрать все")
        self.clear_all_btn = QPushButton("Снять все")
        self.select_all_btn.clicked.connect(lambda: self._set_all_columns(True))
        self.clear_all_btn.clicked.connect(lambda: self._set_all_columns(False))
        btn_row.addWidget(self.select_all_btn)
        btn_row.addWidget(self.clear_all_btn)
        vbox.addLayout(btn_row)

        self.columns_widget = QWidget()
        self.columns_layout = QVBoxLayout(self.columns_widget)
        self.columns_layout.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self.columns_widget)
        return group

    def _build_params_group(self):
        group = QGroupBox("Параметры SDA")
        grid = QGridLayout(group)

        self.n_clusters_min = QSpinBox(); self.n_clusters_min.setRange(2, 200); self.n_clusters_min.setValue(2)
        self.n_clusters_max = QSpinBox(); self.n_clusters_max.setRange(2, 200); self.n_clusters_max.setValue(20)
        self.k_neighbours_min = QSpinBox(); self.k_neighbours_min.setRange(2, 500); self.k_neighbours_min.setValue(20)
        self.k_neighbours_max = QSpinBox(); self.k_neighbours_max.setRange(2, 500); self.k_neighbours_max.setValue(50)

        self.st1_len_thresholds = QLabel("st1_len_thresholds: 0,20,40,60")
        self.st2_len_thresholds = QLabel("st2_len_thresholds: 40")
        self.scale_cb = QCheckBox("Scale features")

        grid.addWidget(QLabel("n_clusters_min"), 0, 0); grid.addWidget(self.n_clusters_min, 0, 1)
        grid.addWidget(QLabel("n_clusters_max"), 0, 2); grid.addWidget(self.n_clusters_max, 0, 3)
        grid.addWidget(QLabel("k_neighbours_min"), 1, 0); grid.addWidget(self.k_neighbours_min, 1, 1)
        grid.addWidget(QLabel("k_neighbours_max"), 1, 2); grid.addWidget(self.k_neighbours_max, 1, 3)
        grid.addWidget(self.st1_len_thresholds, 2, 0, 1, 2)
        grid.addWidget(self.st2_len_thresholds, 2, 2, 1, 2)
        grid.addWidget(self.scale_cb, 3, 0, 1, 4)
        return group

    def _build_advanced_group(self):
        group = QGroupBox("Advanced")
        group.setCheckable(True)
        group.setChecked(False)
        grid = QGridLayout(group)

        self.st1_dist_rate = QDoubleSpinBox(); self.st1_dist_rate.setRange(0.01, 10.0); self.st1_dist_rate.setValue(0.3)
        self.st2_dist_rate = QDoubleSpinBox(); self.st2_dist_rate.setRange(0.01, 10.0); self.st2_dist_rate.setValue(0.2)
        self.n_edge_clusters_min = QSpinBox(); self.n_edge_clusters_min.setRange(2, 100); self.n_edge_clusters_min.setValue(2)
        self.n_edge_clusters_max = QSpinBox(); self.n_edge_clusters_max.setRange(2, 100); self.n_edge_clusters_max.setValue(15)
        self.random_state = QSpinBox(); self.random_state.setRange(0, 999999); self.random_state.setValue(42)
        self.verbose_cb = QCheckBox("verbose"); self.verbose_cb.setChecked(False)

        grid.addWidget(QLabel("st1_dist_rate"), 0, 0); grid.addWidget(self.st1_dist_rate, 0, 1)
        grid.addWidget(QLabel("st2_dist_rate"), 0, 2); grid.addWidget(self.st2_dist_rate, 0, 3)
        grid.addWidget(QLabel("n_edge_clusters_min"), 1, 0); grid.addWidget(self.n_edge_clusters_min, 1, 1)
        grid.addWidget(QLabel("n_edge_clusters_max"), 1, 2); grid.addWidget(self.n_edge_clusters_max, 1, 3)
        grid.addWidget(QLabel("random_state"), 2, 0); grid.addWidget(self.random_state, 2, 1)
        grid.addWidget(self.verbose_cb, 2, 2, 1, 2)
        return group

    def _build_actions(self):
        row = QHBoxLayout()
        self.run_btn = QPushButton("Запустить SDA")
        self.cancel_btn = QPushButton("Отменить")
        self.cancel_btn.setEnabled(False)
        self.reset_btn = QPushButton("Сбросить результат")
        self.export_btn = QPushButton("Экспорт сегментов CSV")

        self.run_btn.clicked.connect(self._run_segmentation)
        self.cancel_btn.clicked.connect(self._cancel_task)
        self.reset_btn.clicked.connect(self._reset_result)
        self.export_btn.clicked.connect(self._export_segments)

        row.addWidget(self.run_btn)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.reset_btn)
        row.addWidget(self.export_btn)
        return row

    def _build_results_group(self):
        group = QGroupBox("Результаты")
        vbox = QVBoxLayout(group)

        self.summary_label = QLabel("Результаты еще не рассчитаны.")
        vbox.addWidget(self.summary_label)

        vbox.addWidget(QLabel("Таблица вариантов SDA"))
        self.results_table = QTableView()
        self.results_table.setMinimumHeight(180)
        vbox.addWidget(self.results_table)

        vbox.addWidget(QLabel("Таблица сегментов"))
        self.segments_table = QTableView()
        self.segments_table.setMinimumHeight(180)
        vbox.addWidget(self.segments_table)

        self.figure = Figure(figsize=(8, 3), constrained_layout=True)
        self.canvas = ScrollFriendlyCanvas(self.figure)
        self.canvas.setMinimumHeight(250)
        vbox.addWidget(self.canvas)

        return group

    def _render_columns(self, columns: List[str]):
        for cb in self.column_checkboxes:
            self.columns_layout.removeWidget(cb)
            cb.deleteLater()
        self.column_checkboxes.clear()

        if not columns:
            self.columns_layout.addWidget(QLabel("Числовые признаки не найдены."))
            return

        for col in columns:
            cb = QCheckBox(col)
            cb.setChecked(True)
            self.columns_layout.addWidget(cb)
            self.column_checkboxes.append(cb)

    def _set_all_columns(self, state: bool):
        for cb in self.column_checkboxes:
            cb.setChecked(state)

    def _collect_params(self):
        return {
            "n_clusters_min": self.n_clusters_min.value(),
            "n_clusters_max": self.n_clusters_max.value(),
            "k_neighbours_min": self.k_neighbours_min.value(),
            "k_neighbours_max": self.k_neighbours_max.value(),
            "st1_len_thresholds": [0, 20, 40, 60],
            "st2_len_thresholds": [40],
            "st1_dist_rate": self.st1_dist_rate.value(),
            "st2_dist_rate": self.st2_dist_rate.value(),
            "n_cl_max_thr": [10, 15, 20],
            "k_neighb_max_thr": [35, 40, 45, 50],
            "n_edge_clusters_min": self.n_edge_clusters_min.value(),
            "n_edge_clusters_max": self.n_edge_clusters_max.value(),
            "random_state": self.random_state.value(),
            "scale": self.scale_cb.isChecked(),
            "verbose": self.verbose_cb.isChecked(),
        }

    def _run_segmentation(self):
        selected_columns = [cb.text() for cb in self.column_checkboxes if cb.isChecked()]
        if not selected_columns:
            self._show_error("Выберите хотя бы один признак.")
            return

        params = self._collect_params()
        request = self.vm.build_segmentation_request(selected_columns, params)

        self._set_busy(True)
        self.status_label.setText("Статус: выполняется сегментация")
        self.progress_bar.setValue(0)

        self.current_worker = PipelineWorker(self.vm.execute_segmentation, **request)
        self.current_worker.signals.progress.connect(self._on_progress)
        self.current_worker.signals.result.connect(self._on_worker_result)
        self.current_worker.signals.error.connect(self._on_worker_error)
        self.current_worker.signals.finished.connect(self._on_worker_finished)
        self.thread_pool.start(self.current_worker)

    def _cancel_task(self):
        if self.current_worker is not None:
            self.current_worker.cancel()
            self.status_label.setText("Статус: отмена...")

    def _on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(f"Статус: {message}")

    def _on_worker_result(self, result):
        self.vm.apply_segmentation_result(result)

    def _on_worker_error(self, message):
        if "отменена" in message.lower():
            self.status_label.setText("Статус: задача отменена")
            return
        self._show_error(message)

    def _on_worker_finished(self, cancelled):
        self._set_busy(False)
        if cancelled:
            self.status_label.setText("Статус: задача отменена")
        elif self.progress_bar.value() < 100:
            self.progress_bar.setValue(100)
            self.status_label.setText("Статус: выполнено")
        self.current_worker = None

    def _set_busy(self, busy: bool):
        self.run_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)

    def _show_result(self, result):
        self.current_result = result
        sorted_results = result.results_table.sort_values(by="Avg-Silh", ascending=False) if "Avg-Silh" in result.results_table.columns else result.results_table
        self.results_table.setModel(DataFrameModel(sorted_results.reset_index(drop=True)))
        self.segments_table.setModel(DataFrameModel(result.segments_table.reset_index(drop=True).fillna("")))

        self.summary_label.setText(
            f"Сегментов: {result.summary.get('n_segments', 0)} | "
            f"Границ: {result.summary.get('n_boundaries', 0)} | "
            f"Avg-Silh: {result.summary.get('Avg-Silh', 'n/a')}"
        )
        self._draw_plot(result)

    def _draw_plot(self, result):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        plot_col = result.selected_columns[0]
        x = list(range(len(result.segmented_data)))
        y = result.segmented_data[plot_col].values
        ax.plot(x, y, linewidth=1.2, label=plot_col)

        for edge in result.edges:
            ax.axvline(edge, color="red", linestyle="--", alpha=0.6)

        ax.set_title("Сегментация SDA")
        ax.set_xlabel("Индекс")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        self.canvas.draw_idle()

    def _reset_result(self):
        self.vm.reset_result()
        self.current_result = None
        self.results_table.setModel(None)
        self.segments_table.setModel(None)
        self.summary_label.setText("Результаты еще не рассчитаны.")
        self.figure.clear()
        self.canvas.draw_idle()

    def _export_segments(self):
        if self.current_result is None:
            self._show_error("Нет результатов для экспорта.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт сегментов",
            str(Path.home() / "segments.csv"),
            "CSV Files (*.csv)",
        )
        if not file_path:
            return

        self.current_result.segments_table.to_csv(file_path, index=False)
        self._show_info(f"Экспортировано: {file_path}")

    def _show_error(self, message: str):
        QMessageBox.critical(self, "Ошибка", message)

    def _show_info(self, message: str):
        QMessageBox.information(self, "Информация", message)
