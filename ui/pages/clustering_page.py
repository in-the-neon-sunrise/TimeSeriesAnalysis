from __future__ import annotations

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PySide6.QtCore import QThreadPool
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
    QProgressBar,
)

from ui.canvas import ScrollFriendlyCanvas
from ui.models.dataframe_model import DataFrameModel
from ui.pages.base_page import BasePage
from viewmodels.clustering_vm import ClusteringViewModel
from workers.pipeline_worker import PipelineWorker


class ClusteringPage(BasePage):
    def __init__(self, data_vm):
        super().__init__()
        self.vm = ClusteringViewModel(data_vm.project)
        self.current_result = None
        self.column_checkboxes: List[QCheckBox] = []
        self.thread_pool = QThreadPool.globalInstance()
        self.current_worker = None
        self._selected_input_key = None
        self._output_name = ""
        self._output_edited = False

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
        self.status_label = QLabel("Статус: ожидание")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(self._build_actions())
        layout.addWidget(self._build_results_group())
        layout.addStretch()

        root = QVBoxLayout(self)
        root.addWidget(scroll)

    def on_enter(self):
        self.vm.refresh()

    def get_dataset_toolbar_state(self):
        options = self._dataset_options()
        if not options:
            return None
        if self._selected_input_key not in [k for k, _, _ in options]:
            self._selected_input_key = options[-1][0]
            self._output_edited = False
        if not self._output_edited:
            self._output_name = f"clusters_from_{self._selected_input_key}"
        from ui.dataset_toolbar import DatasetOption
        return {"options": [DatasetOption(k, t) for k, t, _ in options], "selected_key": self._selected_input_key,
                "output_name": self._output_name}

    def on_toolbar_input_changed(self, key):
        self._selected_input_key = key
        if not self._output_edited:
            self._output_name = f"clusters_from_{key}"

    def on_toolbar_output_changed(self, text):
        self._output_name = text.strip()
        self._output_edited = True

    def _dataset_options(self):
        out = []
        p = getattr(self, "data_vm", None)
        pr = self.vm.project
        if pr.raw_data is not None and not pr.raw_data.empty: out.append(("raw", "исходные данные", pr.raw_data))
        if pr.processed_data is not None and not pr.processed_data.empty: out.append(
            ("processed", "предобработанные данные", pr.processed_data))
        if pr.features is not None and not pr.features.empty: out.append(("features", "признаки", pr.features))
        if pr.segments is not None and not pr.segments.empty: out.append(("segments", "сегменты", pr.segments))
        if pr.clusters is not None and not pr.clusters.empty: out.append(("clusters", "кластеры", pr.clusters))
        return out

    def _build_input_group(self):
        group = QGroupBox("Входные данные: сегменты")
        vbox = QVBoxLayout(group)

        self.source_status = QLabel("Нет данных сегментации.")
        self.input_hint = QLabel(
            "Каждая строка таблицы соответствует одному сегменту. Кластеризация назначает cluster_id каждому сегменту.")
        self.segments_count_label = QLabel("Количество сегментов: 0")
        vbox.addWidget(self.source_status)
        vbox.addWidget(self.segments_count_label)
        vbox.addWidget(self.input_hint)

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
        self.cancel_btn = QPushButton("Отменить")
        self.cancel_btn.setEnabled(False)
        self.reset_btn = QPushButton("Сбросить результат")
        self.export_btn = QPushButton("Экспорт clustered segments CSV")

        self.run_btn.clicked.connect(self._run_clustering)
        self.cancel_btn.clicked.connect(self._cancel_task)
        self.reset_btn.clicked.connect(self.vm.reset_result)
        self.export_btn.clicked.connect(self._export_result)

        row.addWidget(self.run_btn)
        row.addWidget(self.cancel_btn)
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

        self.figure = Figure(figsize=(8, 7), constrained_layout=True)

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

        request = self.vm.build_clustering_request(
            method=self.method_combo.currentData(),
            selected_columns=selected_columns,
            params=self._collect_params(),
        )
        self._set_busy(True)
        self.status_label.setText("Статус: выполняется кластеризация")
        self.progress_bar.setValue(0)

        self.current_worker = PipelineWorker(self.vm.execute_clustering, **request)
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
        self.vm.apply_clustering_result(result)

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
        self._selected_input_key = None
        self._output_name = ""
        self._output_edited = False

    def _set_busy(self, busy: bool):
        self.run_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)

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

        self._draw_scatter()

    def _draw_scatter(self):
        self.figure.clear()
        ax = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)

        if self.current_result is None:
            ax.text(0.5, 0.5, "Нет результата для визуализации", ha="center", va="center")
            ax.axis("off")
            self.canvas.draw_idle()
            return

        cols = list(self.current_result.selected_columns)
        if len(cols) < 2 or len(self.current_result.clustered_segments) < 2:
            ax.text(0.5, 0.5, "Недостаточно сегментов/признаков для PCA", ha="center", va="center")
            ax.axis("off")
        else:
            pca_df = self.vm.clustering_service.build_pca_projection(self.current_result.clustered_segments, cols)
            if pca_df.empty:
                ax.text(0.5, 0.5, "Недостаточно данных для PCA", ha="center", va="center")
                ax.axis("off")
            else:
                labels = pca_df["cluster_id"].to_numpy()
                for label in sorted(set(labels.tolist())):
                    m = labels == label
                    ax.scatter(pca_df.loc[m, "PC1"], pca_df.loc[m, "PC2"], s=28, alpha=0.8,
                               label=("noise (-1)" if label == -1 else f"cluster {label}"))
                ax.set_xlabel("PC1");
                ax.set_ylabel("PC2");
                ax.set_title("Сегменты в пространстве PCA");
                ax.grid(alpha=0.3);
                ax.legend(loc="best")

        fi = self.vm.clustering_service.build_feature_importance(self.current_result.clustered_segments, cols, top_n=10)
        if fi.empty:
            ax2.text(0.5, 0.5, "Невозможно оценить отличающие признаки: найден только один кластер.", ha="center",
                     va="center")
            ax2.axis("off")
        else:
            ax2.barh(fi["feature"], fi["score"], color="#4477AA")
            ax2.invert_yaxis()
            ax2.set_xlabel("Score")
            ax2.set_title("Признаки, наиболее отличающие кластеры")
            ax2.grid(alpha=0.3, axis="x")
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
