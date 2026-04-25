from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ui.canvas import ScrollFriendlyCanvas
from ui.models.dataframe_model import DataFrameModel
from ui.pages.base_page import BasePage
from viewmodels.clustering_vm import ClusteringViewModel


class ClusteringPage(BasePage):
    def __init__(self, data_vm):
        super().__init__()
        self.vm = ClusteringViewModel(data_vm.project)
        self.current_result = None
        self.column_checkboxes: List[QCheckBox] = []

        self.vm.source_info_ready.connect(self._render_source_info)
        self.vm.columns_ready.connect(self._render_columns)
        self.vm.clustering_ready.connect(self._show_result)
        self.vm.result_reset.connect(self._clear_results)
        self.vm.error_occurred.connect(self._show_error)
        self.vm.info_changed.connect(self._show_info)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setSpacing(12)

        title = QLabel("Кластеризация сегментов")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_algorithm_group())
        layout.addWidget(self._build_params_group())
        layout.addLayout(self._build_actions())
        layout.addWidget(self._build_results_group())
        layout.addStretch()

        root = QVBoxLayout(self)
        root.addWidget(scroll)

    def on_enter(self):
        self.vm.refresh()

    def _build_input_group(self):
        group = QGroupBox("Входные данные")
        vbox = QVBoxLayout(group)

        self.source_status = QLabel("Нет данных сегментации.")
        self.segments_count_label = QLabel("Количество сегментов: 0")
        vbox.addWidget(self.source_status)
        vbox.addWidget(self.segments_count_label)

        row = QHBoxLayout()
        self.select_all_btn = QPushButton("Выбрать все")
        self.clear_all_btn = QPushButton("Снять все")
        self.select_all_btn.clicked.connect(lambda: self._set_all_columns(True))
        self.clear_all_btn.clicked.connect(lambda: self._set_all_columns(False))
        row.addWidget(self.select_all_btn)
        row.addWidget(self.clear_all_btn)
        vbox.addLayout(row)

        self.columns_widget = QWidget()
        self.columns_layout = QVBoxLayout(self.columns_widget)
        self.columns_layout.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(self.columns_widget)
        return group

    def _build_algorithm_group(self):
        group = QGroupBox("Алгоритм")
        grid = QGridLayout(group)

        self.method_combo = QComboBox()
        self.method_combo.addItem("KMeans", userData="kmeans")
        self.method_combo.addItem("DBSCAN", userData="dbscan")
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)

        grid.addWidget(QLabel("Метод"), 0, 0)
        grid.addWidget(self.method_combo, 0, 1)
        return group

    def _build_params_group(self):
        group = QGroupBox("Параметры")
        vbox = QVBoxLayout(group)

        self.params_stack = QStackedWidget()
        self.kmeans_widget = self._build_kmeans_params()
        self.dbscan_widget = self._build_dbscan_params()
        self.params_stack.addWidget(self.kmeans_widget)
        self.params_stack.addWidget(self.dbscan_widget)

        vbox.addWidget(self.params_stack)
        return group

    def _build_kmeans_params(self):
        widget = QWidget()
        grid = QGridLayout(widget)

        self.kmeans_n_clusters = QSpinBox(); self.kmeans_n_clusters.setRange(2, 500); self.kmeans_n_clusters.setValue(4)
        self.kmeans_random_state = QSpinBox(); self.kmeans_random_state.setRange(0, 1_000_000); self.kmeans_random_state.setValue(42)
        self.kmeans_n_init = QSpinBox(); self.kmeans_n_init.setRange(1, 200); self.kmeans_n_init.setValue(10)
        self.kmeans_scale = QCheckBox("Scale data")

        grid.addWidget(QLabel("n_clusters"), 0, 0); grid.addWidget(self.kmeans_n_clusters, 0, 1)
        grid.addWidget(QLabel("random_state"), 1, 0); grid.addWidget(self.kmeans_random_state, 1, 1)
        grid.addWidget(QLabel("n_init"), 2, 0); grid.addWidget(self.kmeans_n_init, 2, 1)
        grid.addWidget(self.kmeans_scale, 3, 0, 1, 2)
        return widget

    def _build_dbscan_params(self):
        widget = QWidget()
        grid = QGridLayout(widget)

        self.dbscan_eps = QDoubleSpinBox(); self.dbscan_eps.setDecimals(3); self.dbscan_eps.setRange(0.001, 1000.0); self.dbscan_eps.setValue(0.5)
        self.dbscan_min_samples = QSpinBox(); self.dbscan_min_samples.setRange(1, 500); self.dbscan_min_samples.setValue(5)
        self.dbscan_metric = QComboBox(); self.dbscan_metric.addItems(["euclidean", "manhattan", "cosine"])
        self.dbscan_scale = QCheckBox("Scale data")

        grid.addWidget(QLabel("eps"), 0, 0); grid.addWidget(self.dbscan_eps, 0, 1)
        grid.addWidget(QLabel("min_samples"), 1, 0); grid.addWidget(self.dbscan_min_samples, 1, 1)
        grid.addWidget(QLabel("metric"), 2, 0); grid.addWidget(self.dbscan_metric, 2, 1)
        grid.addWidget(self.dbscan_scale, 3, 0, 1, 2)
        return widget

    def _build_actions(self):
        row = QHBoxLayout()
        self.run_btn = QPushButton("Запустить кластеризацию")
        self.reset_btn = QPushButton("Сбросить результат")
        self.export_btn = QPushButton("Экспорт clustered segments CSV")

        self.run_btn.clicked.connect(self._run_clustering)
        self.reset_btn.clicked.connect(self.vm.reset_result)
        self.export_btn.clicked.connect(self._export_result)

        row.addWidget(self.run_btn)
        row.addWidget(self.reset_btn)
        row.addWidget(self.export_btn)
        return row

    def _build_results_group(self):
        group = QGroupBox("Результаты")
        vbox = QVBoxLayout(group)

        self.summary_label = QLabel("Результаты еще не рассчитаны.")
        self.metrics_label = QLabel("Метрики: n/a")
        vbox.addWidget(self.summary_label)
        vbox.addWidget(self.metrics_label)

        self.results_table = QTableView()
        self.results_table.setMinimumHeight(220)
        vbox.addWidget(self.results_table)

        axes_row = QHBoxLayout()
        self.plot_x_combo = QComboBox()
        self.plot_y_combo = QComboBox()
        self.plot_x_combo.currentIndexChanged.connect(self._draw_scatter)
        self.plot_y_combo.currentIndexChanged.connect(self._draw_scatter)
        axes_row.addWidget(QLabel("X")); axes_row.addWidget(self.plot_x_combo)
        axes_row.addWidget(QLabel("Y")); axes_row.addWidget(self.plot_y_combo)
        vbox.addLayout(axes_row)

        self.figure = Figure(figsize=(8, 4), constrained_layout=True)
        self.canvas = ScrollFriendlyCanvas(self.figure)
        self.canvas.setMinimumHeight(280)
        vbox.addWidget(self.canvas)

        return group

    def _on_method_changed(self):
        method = self.method_combo.currentData()
        self.params_stack.setCurrentIndex(0 if method == "kmeans" else 1)

    def _render_source_info(self, info: dict):
        self.source_status.setText(info.get("message", ""))
        self.segments_count_label.setText(f"Количество сегментов: {info.get('segments_count', 0)}")

    def _render_columns(self, columns: List[str]):
        for cb in self.column_checkboxes:
            self.columns_layout.removeWidget(cb)
            cb.deleteLater()
        self.column_checkboxes.clear()

        if not columns:
            self.columns_layout.addWidget(QLabel("Числовые признаки сегментов недоступны."))
            return

        for col in columns:
            cb = QCheckBox(col)
            cb.setChecked(True)
            self.columns_layout.addWidget(cb)
            self.column_checkboxes.append(cb)

        self.plot_x_combo.clear()
        self.plot_y_combo.clear()
        self.plot_x_combo.addItems(columns)
        self.plot_y_combo.addItems(columns)
        if len(columns) > 1:
            self.plot_y_combo.setCurrentIndex(1)

    def _set_all_columns(self, state: bool):
        for cb in self.column_checkboxes:
            cb.setChecked(state)

    def _collect_selected_columns(self) -> List[str]:
        return [cb.text() for cb in self.column_checkboxes if cb.isChecked()]

    def _collect_params(self):
        method = self.method_combo.currentData()
        if method == "kmeans":
            return {
                "n_clusters": self.kmeans_n_clusters.value(),
                "random_state": self.kmeans_random_state.value(),
                "n_init": self.kmeans_n_init.value(),
                "scale": self.kmeans_scale.isChecked(),
            }

        return {
            "eps": self.dbscan_eps.value(),
            "min_samples": self.dbscan_min_samples.value(),
            "metric": self.dbscan_metric.currentText(),
            "scale": self.dbscan_scale.isChecked(),
        }

    def _run_clustering(self):
        selected_columns = self._collect_selected_columns()
        if not selected_columns:
            self._show_error("Выберите хотя бы один числовой признак.")
            return

        self.vm.run_clustering(
            method=self.method_combo.currentData(),
            selected_columns=selected_columns,
            params=self._collect_params(),
        )

    def _show_result(self, result):
        self.current_result = result
        self.results_table.setModel(DataFrameModel(result.clustered_segments.reset_index(drop=True).fillna("")))

        metric_text = []
        for key, value in result.metrics.items():
            metric_text.append(f"{key}: {'n/a' if value is None else f'{value:.4f}'}")

        summary = result.summary
        self.summary_label.setText(
            " | ".join(
                [
                    f"method={summary.get('method')}",
                    f"segments={summary.get('number_of_segments', 0)}",
                    f"clusters={summary.get('number_of_clusters', 0)}",
                    f"noise={summary.get('number_of_noise_points', 0)}",
                ]
            )
        )
        self.metrics_label.setText("Метрики: " + "; ".join(metric_text))

        self.plot_x_combo.clear()
        self.plot_y_combo.clear()
        self.plot_x_combo.addItems(result.selected_columns)
        self.plot_y_combo.addItems(result.selected_columns)
        if len(result.selected_columns) > 1:
            self.plot_y_combo.setCurrentIndex(1)

        self._draw_scatter()

    def _draw_scatter(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if self.current_result is None:
            ax.text(0.5, 0.5, "Нет результата для визуализации", ha="center", va="center")
            ax.axis("off")
            self.canvas.draw_idle()
            return

        cols = list(self.current_result.selected_columns)
        if len(cols) < 2:
            ax.text(0.5, 0.5, "Scatter plot требует минимум два признака", ha="center", va="center")
            ax.axis("off")
            self.canvas.draw_idle()
            return

        x_col = self.plot_x_combo.currentText() or cols[0]
        y_col = self.plot_y_combo.currentText() or cols[1]

        df = self.current_result.clustered_segments
        if x_col not in df.columns or y_col not in df.columns:
            ax.text(0.5, 0.5, "Выбранные признаки недоступны", ha="center", va="center")
            ax.axis("off")
            self.canvas.draw_idle()
            return

        labels = df["cluster_id"].to_numpy()
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)

        x = x[mask]
        y = y[mask]
        labels = labels[mask]

        unique_labels = sorted(set(labels.tolist()))
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_name = "noise (-1)" if label == -1 else f"cluster {label}"
            ax.scatter(x[cluster_mask], y[cluster_mask], s=26, alpha=0.8, label=cluster_name)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title("Кластеры сегментов")
        ax.grid(alpha=0.3)
        if unique_labels:
            ax.legend(loc="best")
        self.canvas.draw_idle()

    def _export_result(self):
        if self.current_result is None:
            self._show_error("Нет результатов для экспорта.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт clustered segments",
            str(Path.home() / "clustered_segments.csv"),
            "CSV Files (*.csv)",
        )
        if not file_path:
            return

        try:
            self.vm.export_clustered_segments(file_path)
            self._show_info(f"Экспортировано: {file_path}")
        except Exception as exc:
            self._show_error(str(exc))

    def _clear_results(self):
        self.current_result = None
        self.summary_label.setText("Результаты еще не рассчитаны.")
        self.metrics_label.setText("Метрики: n/a")
        self.results_table.setModel(None)
        self.figure.clear()
        self.canvas.draw_idle()

    def _show_error(self, message: str):
        QMessageBox.critical(self, "Ошибка", message)

    def _show_info(self, message: str):
        QMessageBox.information(self, "Информация", message)
