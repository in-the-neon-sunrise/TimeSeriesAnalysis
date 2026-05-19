from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.canvas import ScrollFriendlyCanvas
from ui.models.dataframe_model import DataFrameModel
from ui.pages.base_page import BasePage
from viewmodels.clustering_vm import ClusteringViewModel
from workers.pipeline_worker import PipelineWorker


class ClusteringPage(BasePage):
    """
    Страница кластеризации сегментов.

    Логика:
    - входом является таблица сегментов, а не построчные данные;
    - выбор признаков организован двумерной таблицей: исходный признак x агрегат;
    - для KMeans доступен подбор k по метрикам;
    - графики разделены на отдельные высокие блоки, а не сжаты в один subplot.
    """

    def __init__(self, data_vm):
        super().__init__()
        self.vm = ClusteringViewModel(data_vm.project)
        self.current_result = None
        self.k_evaluation_df: pd.DataFrame | None = None
        self.feature_column_map: Dict[Tuple[str, str], str] = {}
        self.feature_suffixes: List[str] = []

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
        layout.addWidget(self._build_k_selection_group())

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

    # =========================
    # Page lifecycle / toolbar
    # =========================

    def on_enter(self):
        self.vm.refresh()

    def get_dataset_toolbar_state(self):
        options = self._dataset_options()
        if not options:
            return None

        if self._selected_input_key not in [k for k, _, _ in options]:
            self._selected_input_key = "segments"
            self._output_edited = False

        if not self._output_edited:
            self._output_name = f"clusters_from_{self._selected_input_key}"

        from ui.dataset_toolbar import DatasetOption
        return {
            "options": [DatasetOption(k, t) for k, t, _ in options],
            "selected_key": self._selected_input_key,
            "output_name": self._output_name,
        }

    def on_toolbar_input_changed(self, key):
        self._selected_input_key = key
        if not self._output_edited:
            self._output_name = f"clusters_from_{key}"

    def on_toolbar_output_changed(self, text):
        self._output_name = text.strip()
        self._output_edited = True

    def _dataset_options(self):
        pr = self.vm.project
        out = []
        if pr.segments is not None and not pr.segments.empty:
            out.append(("segments", "сегменты", pr.segments))
        return out

    # =========================
    # UI builders
    # =========================

    def _build_input_group(self):
        group = QGroupBox("Входные данные: сегменты")
        vbox = QVBoxLayout(group)

        self.source_status = QLabel("Нет данных сегментации.")
        self.input_hint = QLabel(
            "Каждая строка таблицы соответствует одному сегменту. "
            "Кластеризация назначает cluster_id каждому сегменту, а не каждой строке исходного временного ряда."
        )
        self.input_hint.setWordWrap(True)
        self.segments_count_label = QLabel("Количество сегментов: 0")

        vbox.addWidget(self.source_status)
        vbox.addWidget(self.segments_count_label)
        vbox.addWidget(self.input_hint)

        controls = QHBoxLayout()
        self.select_recommended_btn = QPushButton("Рекомендованный набор")
        self.select_mean_std_btn = QPushButton("Mean + Std")
        self.select_all_btn = QPushButton("Выбрать все")
        self.clear_all_btn = QPushButton("Снять все")

        self.select_recommended_btn.clicked.connect(self._select_recommended_features)
        self.select_mean_std_btn.clicked.connect(lambda: self._select_suffixes({"mean", "std"}))
        self.select_all_btn.clicked.connect(lambda: self._set_all_features(True))
        self.clear_all_btn.clicked.connect(lambda: self._set_all_features(False))

        controls.addWidget(self.select_recommended_btn)
        controls.addWidget(self.select_mean_std_btn)
        controls.addWidget(self.select_all_btn)
        controls.addWidget(self.clear_all_btn)
        vbox.addLayout(controls)

        self.features_table = QTableWidget()
        self.features_table.setMinimumHeight(260)
        self.features_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.features_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.features_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        vbox.addWidget(self.features_table)

        self.selected_features_label = QLabel("Выбрано признаков: 0")
        vbox.addWidget(self.selected_features_label)

        return group

    def _build_algorithm_group(self):
        group = QGroupBox("Алгоритм")
        grid = QGridLayout(group)

        self.method_combo = QComboBox()
        self.method_combo.addItem("KMeans", userData="kmeans")
        self.method_combo.addItem("DBSCAN", userData="dbscan")
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)

        method_hint = QLabel(
            "Для основной демонстрации лучше использовать KMeans: он стабильнее, понятнее для защиты "
            "и поддерживает подбор числа кластеров. DBSCAN оставлен как дополнительный вариант."
        )
        method_hint.setWordWrap(True)

        grid.addWidget(QLabel("Метод"), 0, 0)
        grid.addWidget(self.method_combo, 0, 1)
        grid.addWidget(method_hint, 1, 0, 1, 2)
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

        self.kmeans_n_clusters = QSpinBox()
        self.kmeans_n_clusters.setRange(2, 500)
        self.kmeans_n_clusters.setValue(3)

        self.kmeans_scale = QCheckBox("Масштабировать признаки")
        self.kmeans_scale.setChecked(True)

        self.kmeans_advanced = QGroupBox("Advanced")
        self.kmeans_advanced.setCheckable(True)
        self.kmeans_advanced.setChecked(False)
        adv = QGridLayout(self.kmeans_advanced)

        self.kmeans_random_state = QSpinBox()
        self.kmeans_random_state.setRange(0, 1_000_000)
        self.kmeans_random_state.setValue(42)

        self.kmeans_n_init = QSpinBox()
        self.kmeans_n_init.setRange(1, 200)
        self.kmeans_n_init.setValue(10)

        adv.addWidget(QLabel("random_state"), 0, 0)
        adv.addWidget(self.kmeans_random_state, 0, 1)
        adv.addWidget(QLabel("n_init"), 1, 0)
        adv.addWidget(self.kmeans_n_init, 1, 1)

        grid.addWidget(QLabel("n_clusters"), 0, 0)
        grid.addWidget(self.kmeans_n_clusters, 0, 1)
        grid.addWidget(self.kmeans_scale, 1, 0, 1, 2)
        grid.addWidget(self.kmeans_advanced, 2, 0, 1, 2)
        return widget

    def _build_dbscan_params(self):
        widget = QWidget()
        grid = QGridLayout(widget)

        self.dbscan_eps = QDoubleSpinBox()
        self.dbscan_eps.setDecimals(3)
        self.dbscan_eps.setRange(0.001, 1000.0)
        self.dbscan_eps.setValue(0.5)

        self.dbscan_min_samples = QSpinBox()
        self.dbscan_min_samples.setRange(1, 500)
        self.dbscan_min_samples.setValue(3)

        self.dbscan_metric = QComboBox()
        self.dbscan_metric.addItems(["euclidean", "manhattan", "cosine"])

        self.dbscan_scale = QCheckBox("Масштабировать признаки")
        self.dbscan_scale.setChecked(True)

        grid.addWidget(QLabel("eps"), 0, 0)
        grid.addWidget(self.dbscan_eps, 0, 1)
        grid.addWidget(QLabel("min_samples"), 1, 0)
        grid.addWidget(self.dbscan_min_samples, 1, 1)
        grid.addWidget(QLabel("metric"), 2, 0)
        grid.addWidget(self.dbscan_metric, 2, 1)
        grid.addWidget(self.dbscan_scale, 3, 0, 1, 2)
        return widget

    def _build_k_selection_group(self):
        group = QGroupBox("Подбор числа кластеров для KMeans")
        grid = QGridLayout(group)

        self.k_min_spin = QSpinBox()
        self.k_min_spin.setRange(2, 500)
        self.k_min_spin.setValue(2)

        self.k_max_spin = QSpinBox()
        self.k_max_spin.setRange(2, 500)
        self.k_max_spin.setValue(10)

        self.k_metric_combo = QComboBox()
        self.k_metric_combo.addItem("Silhouette — больше лучше", "silhouette")
        self.k_metric_combo.addItem("Calinski-Harabasz — больше лучше", "calinski_harabasz")
        self.k_metric_combo.addItem("Davies-Bouldin — меньше лучше", "davies_bouldin")

        self.evaluate_k_btn = QPushButton("Подобрать k")
        self.apply_best_k_btn = QPushButton("Применить лучший k")
        self.apply_best_k_btn.setEnabled(False)

        self.evaluate_k_btn.clicked.connect(self._evaluate_k_range)
        self.apply_best_k_btn.clicked.connect(self._apply_best_k)

        self.best_k_label = QLabel("Лучший k: —")
        self.best_k_label.setWordWrap(True)

        grid.addWidget(QLabel("k min"), 0, 0)
        grid.addWidget(self.k_min_spin, 0, 1)
        grid.addWidget(QLabel("k max"), 0, 2)
        grid.addWidget(self.k_max_spin, 0, 3)
        grid.addWidget(QLabel("Метрика выбора"), 1, 0)
        grid.addWidget(self.k_metric_combo, 1, 1, 1, 3)
        grid.addWidget(self.evaluate_k_btn, 2, 0, 1, 2)
        grid.addWidget(self.apply_best_k_btn, 2, 2, 1, 2)
        grid.addWidget(self.best_k_label, 3, 0, 1, 4)

        return group

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
        self.summary_label.setWordWrap(True)
        vbox.addWidget(self.summary_label)

        self.warnings_label = QLabel("")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setStyleSheet("color: #8a5a00;")
        self.warnings_label.setVisible(False)
        vbox.addWidget(self.warnings_label)

        metrics_group = QGroupBox("Метрики выбранной кластеризации")
        metrics_grid = QGridLayout(metrics_group)
        self.metric_labels = {}
        for i, (key, title) in enumerate(
            [
                ("silhouette", "Silhouette"),
                ("calinski_harabasz", "Calinski-Harabasz"),
                ("davies_bouldin", "Davies-Bouldin"),
                ("number_of_segments", "Сегментов"),
                ("number_of_clusters", "Кластеров"),
                ("number_of_noise_points", "Noise"),
            ]
        ):
            value_label = QLabel("—")
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.metric_labels[key] = value_label
            row, col = divmod(i, 2)
            metrics_grid.addWidget(QLabel(f"{title}:"), row, col * 2)
            metrics_grid.addWidget(value_label, row, col * 2 + 1)
        vbox.addWidget(metrics_group)

        self.k_curve_group = QGroupBox("График метрики от числа кластеров")
        k_curve_layout = QVBoxLayout(self.k_curve_group)
        self.k_curve_figure = Figure(figsize=(8, 3.5), constrained_layout=True)
        self.k_curve_canvas = ScrollFriendlyCanvas(self.k_curve_figure)
        self.k_curve_canvas.setMinimumHeight(330)
        k_curve_layout.addWidget(self.k_curve_canvas)
        vbox.addWidget(self.k_curve_group)

        self.pca_group = QGroupBox("Сегменты в пространстве PCA")
        pca_layout = QVBoxLayout(self.pca_group)
        self.pca_figure = Figure(figsize=(8, 4.2), constrained_layout=True)
        self.pca_canvas = ScrollFriendlyCanvas(self.pca_figure)
        self.pca_canvas.setMinimumHeight(390)
        pca_layout.addWidget(self.pca_canvas)
        vbox.addWidget(self.pca_group)

        self.timeline_group = QGroupBox("Последовательность кластеров по сегментам")
        timeline_layout = QVBoxLayout(self.timeline_group)
        self.timeline_figure = Figure(figsize=(8, 2.4), constrained_layout=True)
        self.timeline_canvas = ScrollFriendlyCanvas(self.timeline_figure)
        self.timeline_canvas.setMinimumHeight(240)
        timeline_layout.addWidget(self.timeline_canvas)
        vbox.addWidget(self.timeline_group)

        self.importance_group = QGroupBox("Признаки, наиболее отличающие кластеры")
        importance_layout = QVBoxLayout(self.importance_group)
        self.importance_figure = Figure(figsize=(8, 3.4), constrained_layout=True)
        self.importance_canvas = ScrollFriendlyCanvas(self.importance_figure)
        self.importance_canvas.setMinimumHeight(330)
        importance_layout.addWidget(self.importance_canvas)
        vbox.addWidget(self.importance_group)

        self.cluster_sizes_table = QTableView()
        self.cluster_sizes_table.setMinimumHeight(140)
        vbox.addWidget(QLabel("Размеры кластеров"))
        vbox.addWidget(self.cluster_sizes_table)

        self.results_table = QTableView()
        self.results_table.setMinimumHeight(260)
        vbox.addWidget(QLabel("Кластеризованные сегменты"))
        vbox.addWidget(self.results_table)

        return group

    # =========================
    # Source / feature table
    # =========================

    def _render_source_info(self, info: dict):
        self.source_status.setText(info.get("message", ""))
        self.segments_count_label.setText(f"Количество сегментов: {info.get('segments_count', 0)}")

        n = int(info.get("segments_count", 0) or 0)
        max_k = max(2, min(10, n - 1)) if n > 2 else 2
        self.k_max_spin.setValue(max_k)
        self.kmeans_n_clusters.setMaximum(max(2, n))
        self.k_min_spin.setMaximum(max(2, n))
        self.k_max_spin.setMaximum(max(2, n))

    def _render_columns(self, columns: List[str]):
        self._build_feature_checkbox_table(columns)
        self._update_selected_features_label()

    @staticmethod
    def _split_feature_name(column: str) -> Tuple[str, str]:
        if column == "length":
            return "segment", "length"

        known_suffixes = ("mean", "std", "min", "max", "median", "var", "variance", "skew", "kurtosis", "energy", "rms")
        for suffix in known_suffixes:
            marker = f"_{suffix}"
            if column.endswith(marker):
                return column[: -len(marker)], suffix
        return column, "value"

    def _build_feature_checkbox_table(self, columns: List[str]):
        self.feature_column_map.clear()

        grouped: Dict[str, Dict[str, str]] = {}
        suffix_order: List[str] = []

        for col in columns:
            base, suffix = self._split_feature_name(col)
            grouped.setdefault(base, {})[suffix] = col
            if suffix not in suffix_order:
                suffix_order.append(suffix)

        preferred_order = ["length", "mean", "std", "min", "max", "median", "var", "variance", "skew", "kurtosis", "energy", "rms", "value"]
        suffixes = [s for s in preferred_order if s in suffix_order] + [s for s in suffix_order if s not in preferred_order]
        bases = sorted(grouped.keys())

        self.feature_suffixes = suffixes
        self.features_table.clear()
        self.features_table.setRowCount(len(bases))
        self.features_table.setColumnCount(len(suffixes) + 1)
        self.features_table.setHorizontalHeaderLabels(["Признак"] + suffixes)

        for row, base in enumerate(bases):
            name_item = QTableWidgetItem(base)
            name_item.setFlags(Qt.ItemIsEnabled)
            self.features_table.setItem(row, 0, name_item)

            for col_idx, suffix in enumerate(suffixes, start=1):
                original_col = grouped[base].get(suffix)
                if original_col is None:
                    item = QTableWidgetItem("—")
                    item.setFlags(Qt.ItemIsEnabled)
                    self.features_table.setItem(row, col_idx, item)
                    continue

                item = QTableWidgetItem("")
                item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)

                checked = suffix in {"mean", "std"} or original_col == "length"
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)

                item.setToolTip(original_col)
                self.features_table.setItem(row, col_idx, item)
                self.feature_column_map[(base, suffix)] = original_col

        self.features_table.itemChanged.connect(lambda *_: self._update_selected_features_label())
        self.features_table.resizeRowsToContents()

    def _selected_feature_columns(self) -> List[str]:
        selected: List[str] = []
        for row in range(self.features_table.rowCount()):
            base_item = self.features_table.item(row, 0)
            if base_item is None:
                continue
            base = base_item.text()

            for col_idx, suffix in enumerate(self.feature_suffixes, start=1):
                item = self.features_table.item(row, col_idx)
                if item is None:
                    continue
                if item.flags() & Qt.ItemIsUserCheckable and item.checkState() == Qt.Checked:
                    original_col = self.feature_column_map.get((base, suffix))
                    if original_col:
                        selected.append(original_col)

        return selected

    def _set_all_features(self, checked: bool):
        self.features_table.blockSignals(True)
        for row in range(self.features_table.rowCount()):
            for col in range(1, self.features_table.columnCount()):
                item = self.features_table.item(row, col)
                if item is not None and item.flags() & Qt.ItemIsUserCheckable:
                    item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.features_table.blockSignals(False)
        self._update_selected_features_label()

    def _select_suffixes(self, suffixes: set[str]):
        self.features_table.blockSignals(True)
        for row in range(self.features_table.rowCount()):
            for col_idx, suffix in enumerate(self.feature_suffixes, start=1):
                item = self.features_table.item(row, col_idx)
                if item is not None and item.flags() & Qt.ItemIsUserCheckable:
                    item.setCheckState(Qt.Checked if suffix in suffixes else Qt.Unchecked)
        self.features_table.blockSignals(False)
        self._update_selected_features_label()

    def _select_recommended_features(self):
        self.features_table.blockSignals(True)
        for row in range(self.features_table.rowCount()):
            for col_idx, suffix in enumerate(self.feature_suffixes, start=1):
                item = self.features_table.item(row, col_idx)
                if item is None or not (item.flags() & Qt.ItemIsUserCheckable):
                    continue

                original_col = item.toolTip()
                checked = suffix in {"mean", "std"} or original_col == "length"
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.features_table.blockSignals(False)
        self._update_selected_features_label()

    def _update_selected_features_label(self):
        selected = self._selected_feature_columns()
        self.selected_features_label.setText(f"Выбрано признаков: {len(selected)}")

    # =========================
    # Params
    # =========================

    def _on_method_changed(self):
        method = self.method_combo.currentData()
        self.params_stack.setCurrentIndex(0 if method == "kmeans" else 1)
        self.evaluate_k_btn.setEnabled(method == "kmeans")
        self.apply_best_k_btn.setEnabled(method == "kmeans" and self.k_evaluation_df is not None)

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

    # =========================
    # k selection
    # =========================

    def _evaluate_k_range(self):
        if self.method_combo.currentData() != "kmeans":
            self._show_error("Подбор числа кластеров доступен только для KMeans.")
            return

        selected_columns = self._selected_feature_columns()
        if not selected_columns:
            self._show_error("Выберите хотя бы один признак сегментов.")
            return

        if self.k_min_spin.value() > self.k_max_spin.value():
            self._show_error("k min не может быть больше k max.")
            return

        try:
            params = self._collect_params()
            evaluation = self.vm.evaluate_kmeans_range(
                selected_columns=selected_columns,
                params=params,
                k_min=self.k_min_spin.value(),
                k_max=self.k_max_spin.value(),
            )
            self.k_evaluation_df = evaluation

            metric = self.k_metric_combo.currentData()
            best_k = self.vm.clustering_service.choose_best_k(evaluation, metric)
            if best_k is not None:
                self.best_k_label.setText(f"Лучший k по выбранной метрике: {best_k}")
                self.apply_best_k_btn.setEnabled(True)
            else:
                self.best_k_label.setText("Лучший k: не удалось определить")
                self.apply_best_k_btn.setEnabled(False)

            self._draw_k_curve(evaluation)
            self._show_info("Подбор числа кластеров завершен.")
        except Exception as exc:
            self._show_error(str(exc))

    def _apply_best_k(self):
        if self.k_evaluation_df is None or self.k_evaluation_df.empty:
            self._show_error("Сначала выполните подбор k.")
            return

        metric = self.k_metric_combo.currentData()
        best_k = self.vm.clustering_service.choose_best_k(self.k_evaluation_df, metric)
        if best_k is None:
            self._show_error("Не удалось определить лучший k.")
            return

        self.kmeans_n_clusters.setValue(best_k)
        self.best_k_label.setText(f"Применен лучший k: {best_k}")

    # =========================
    # Run / worker
    # =========================

    def _run_clustering(self):
        selected_columns = self._selected_feature_columns()
        if not selected_columns:
            self._show_error("Выберите хотя бы один числовой признак сегмента.")
            return

        request = self.vm.build_clustering_request(
            method=self.method_combo.currentData(),
            selected_columns=selected_columns,
            params=self._collect_params(),
            source_key="segments",
            output_name=self._output_name,
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

    def _set_busy(self, busy: bool):
        self.run_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.evaluate_k_btn.setEnabled(not busy and self.method_combo.currentData() == "kmeans")

    # =========================
    # Result rendering
    # =========================

    @staticmethod
    def _fmt(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    def _show_result(self, result):
        self.current_result = result

        self.results_table.setModel(DataFrameModel(result.clustered_segments.reset_index(drop=True).fillna("")))

        cluster_sizes = self.vm.clustering_service.cluster_size_table(result.clustered_segments)
        self.cluster_sizes_table.setModel(DataFrameModel(cluster_sizes))

        for key, label in self.metric_labels.items():
            if key in result.metrics:
                label.setText(self._fmt(result.metrics.get(key)))
            else:
                label.setText(self._fmt(result.summary.get(key)))

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

        warnings = summary.get("warnings", [])
        if warnings:
            self.warnings_label.setText("Предупреждения:\n" + "\n".join(f"• {w}" for w in warnings))
            self.warnings_label.setVisible(True)
        else:
            self.warnings_label.setVisible(False)

        if self.k_evaluation_df is not None and not self.k_evaluation_df.empty:
            self._draw_k_curve(self.k_evaluation_df)
        else:
            self.k_curve_figure.clear()
            self.k_curve_canvas.draw_idle()

        self._draw_pca(result)
        self._draw_timeline(result)
        self._draw_feature_importance(result)

    def _draw_k_curve(self, evaluation_df: pd.DataFrame):
        self.k_curve_figure.clear()
        ax = self.k_curve_figure.add_subplot(111)

        if evaluation_df is None or evaluation_df.empty:
            ax.text(0.5, 0.5, "Сначала выполните подбор k", ha="center", va="center")
            ax.axis("off")
            self.k_curve_canvas.draw_idle()
            return

        metric = self.k_metric_combo.currentData()
        if metric not in evaluation_df.columns:
            metric = "silhouette"

        ax.plot(evaluation_df["k"], evaluation_df[metric], marker="o")
        ax.set_xlabel("Количество кластеров k")
        ax.set_ylabel(metric)
        ax.set_title(f"Изменение метрики {metric} от числа кластеров")
        ax.grid(alpha=0.3)

        best_k = self.vm.clustering_service.choose_best_k(evaluation_df, metric)
        if best_k is not None:
            best_row = evaluation_df[evaluation_df["k"] == best_k]
            if not best_row.empty:
                ax.axvline(best_k, linestyle="--", alpha=0.6)
                ax.scatter(best_k, best_row.iloc[0][metric], s=80)

        self.k_curve_canvas.draw_idle()

    def _draw_pca(self, result):
        self.pca_figure.clear()
        ax = self.pca_figure.add_subplot(111)

        cols = list(result.selected_columns)
        if len(cols) < 2 or len(result.clustered_segments) < 2:
            ax.text(0.5, 0.5, "Недостаточно сегментов/признаков для PCA", ha="center", va="center")
            ax.axis("off")
            self.pca_canvas.draw_idle()
            return

        pca_df = self.vm.clustering_service.build_pca_projection(result.clustered_segments, cols)
        if pca_df.empty:
            ax.text(0.5, 0.5, "Недостаточно данных для PCA", ha="center", va="center")
            ax.axis("off")
            self.pca_canvas.draw_idle()
            return

        labels = pca_df["cluster_id"].to_numpy()
        for label in sorted(set(labels.tolist())):
            mask = labels == label
            ax.scatter(
                pca_df.loc[mask, "PC1"],
                pca_df.loc[mask, "PC2"],
                s=45,
                alpha=0.85,
                label=("noise (-1)" if label == -1 else f"cluster {label}"),
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Сегменты в пространстве PCA")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        self.pca_canvas.draw_idle()

    def _draw_timeline(self, result):
        self.timeline_figure.clear()
        ax = self.timeline_figure.add_subplot(111)

        df = result.clustered_segments.reset_index(drop=True)
        if df.empty or "cluster_id" not in df.columns:
            ax.text(0.5, 0.5, "Нет cluster_id для отображения", ha="center", va="center")
            ax.axis("off")
            self.timeline_canvas.draw_idle()
            return

        x = list(range(len(df)))
        y = [1] * len(df)
        labels = df["cluster_id"].to_numpy()

        scatter = ax.scatter(x, y, c=labels, s=90, marker="s")
        ax.set_yticks([])
        ax.set_xlabel("Порядок сегментов во времени")
        ax.set_title("Последовательность кластеров по сегментам")
        ax.grid(alpha=0.2, axis="x")

        for i, label in enumerate(labels):
            ax.text(i, 1.06, str(label), ha="center", va="bottom", fontsize=8)

        self.timeline_figure.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.25, label="cluster_id")
        self.timeline_canvas.draw_idle()

    def _draw_feature_importance(self, result):
        self.importance_figure.clear()
        ax = self.importance_figure.add_subplot(111)

        cols = list(result.selected_columns)
        fi = self.vm.clustering_service.build_feature_importance(result.clustered_segments, cols, top_n=12)

        if fi.empty:
            ax.text(
                0.5,
                0.5,
                "Невозможно оценить отличающие признаки: найден только один кластер или мало данных.",
                ha="center",
                va="center",
            )
            ax.axis("off")
            self.importance_canvas.draw_idle()
            return

        ax.barh(fi["feature"], fi["score"])
        ax.invert_yaxis()
        ax.set_xlabel("F-score")
        ax.set_title("Признаки, наиболее отличающие кластеры")
        ax.grid(alpha=0.3, axis="x")
        self.importance_canvas.draw_idle()

    # =========================
    # Export / reset / messages
    # =========================

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
        self.k_evaluation_df = None
        self.summary_label.setText("Результаты еще не рассчитаны.")
        self.warnings_label.setVisible(False)

        for label in getattr(self, "metric_labels", {}).values():
            label.setText("—")

        self.results_table.setModel(None)
        self.cluster_sizes_table.setModel(None)

        for fig, canvas in [
            (self.k_curve_figure, self.k_curve_canvas),
            (self.pca_figure, self.pca_canvas),
            (self.timeline_figure, self.timeline_canvas),
            (self.importance_figure, self.importance_canvas),
        ]:
            fig.clear()
            canvas.draw_idle()

        self.best_k_label.setText("Лучший k: —")
        self.apply_best_k_btn.setEnabled(False)

    def _show_error(self, message: str):
        QMessageBox.critical(self, "Ошибка", message)

    def _show_info(self, message: str):
        self.status_label.setText(f"Статус: {message}")
