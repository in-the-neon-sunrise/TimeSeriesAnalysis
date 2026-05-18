from __future__ import annotations

from pathlib import Path
from typing import List, Any

from matplotlib.figure import Figure
from PySide6.QtCore import QThreadPool, Qt
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
    QProgressBar,
    QComboBox,
    QFrame,
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
        self._selected_input_key = None
        self._output_name = ""
        self._output_edited = False
        self._auto_chunk_defaults_enabled = True

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
        self.vm.load_available_columns(self._selected_input_key)
        self._apply_auto_chunk_defaults()

    def get_dataset_toolbar_state(self):
        options = self._dataset_options()
        if not options:
            return None

        available_keys = [k for k, _, _ in options]
        if self._selected_input_key not in available_keys:
            self._selected_input_key = options[-1][0]
            self._output_edited = False

        if not self._output_edited:
            self._output_name = f"segments_from_{self._selected_input_key}"

        from ui.dataset_toolbar import DatasetOption
        return {
            "options": [DatasetOption(k, t) for k, t, _ in options],
            "selected_key": self._selected_input_key,
            "output_name": self._output_name,
        }

    def on_toolbar_input_changed(self, key):
        self._selected_input_key = key
        if not self._output_edited:
            self._output_name = f"segments_from_{key}"
        self.vm.load_available_columns(self._selected_input_key)
        self._auto_chunk_defaults_enabled = True
        self._apply_auto_chunk_defaults()

    def on_toolbar_output_changed(self, text):
        self._output_name = text.strip()
        self._output_edited = True

    def _dataset_options(self):
        out = []
        pr = self.vm.project

        if pr.raw_data is not None and not pr.raw_data.empty:
            out.append(("raw", "исходные данные", pr.raw_data))
        if pr.processed_data is not None and not pr.processed_data.empty:
            out.append(("processed", "предобработанные данные", pr.processed_data))
        if pr.features is not None and not pr.features.empty:
            out.append(("features", "признаки", pr.features))

        # Эти источники оставлены для совместимости с общей панелью датасетов,
        # но обычно сегментацию нужно запускать по features / processed / raw.
        if pr.segments is not None and not pr.segments.empty:
            out.append(("segments", "сегменты", pr.segments))
        if pr.clusters is not None and not pr.clusters.empty:
            out.append(("clusters", "кластеры", pr.clusters))
        return out

    def _current_input_row_count(self) -> int:
        options = self._dataset_options()
        for key, _, df in options:
            if key == self._selected_input_key:
                return len(df)
        return len(options[-1][2]) if options else 0


    def _build_input_group(self):
        group = QGroupBox("Входные данные")
        vbox = QVBoxLayout(group)

        self.input_source_label = QLabel("Источник выбирается в верхней панели датасетов")
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
        group = QGroupBox("Основные параметры SDA")
        grid = QGridLayout(group)

        self.n_clusters_min = QSpinBox()
        self.n_clusters_min.setRange(2, 200)
        self.n_clusters_min.setValue(2)

        self.n_clusters_max = QSpinBox()
        self.n_clusters_max.setRange(2, 200)
        self.n_clusters_max.setValue(12)

        self.k_neighbours_min = QSpinBox()
        self.k_neighbours_min.setRange(2, 500)
        self.k_neighbours_min.setValue(20)

        self.k_neighbours_max = QSpinBox()
        self.k_neighbours_max.setRange(2, 500)
        self.k_neighbours_max.setValue(50)

        self.scale_cb = QCheckBox("Масштабировать признаки перед SDA")
        self.scale_cb.setChecked(True)

        self.prefer_segments_cb = QCheckBox(
            "Не выбирать слишком простую сегментацию, если есть вариант с числом сегментов не меньше n_clusters_min"
        )
        self.prefer_segments_cb.setChecked(True)
        self.prefer_segments_cb.setToolTip(
            "SDA может объединять стадии после первичного поиска. Эта настройка не запрещает объединение, "
            "но при выборе лучшего варианта сначала предпочитает варианты с достаточным числом сегментов."
        )

        self.cluster_hint_label = QLabel(
            "Важно: n_clusters_min/max задают диапазон перебора кластеров на первом этапе SDA. "
            "Это не жесткое ограничение на итоговое число сегментов, потому что после stage merging сегменты могут объединяться."
        )
        self.cluster_hint_label.setWordWrap(True)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Обработать весь набор данных", "full")
        self.mode_combo.addItem("Обработать по частям", "chunked")

        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(100, 10_000_000)
        self.chunk_size.setValue(1000)

        self.overlap = QSpinBox()
        self.overlap.setRange(0, 1_000_000)
        self.overlap.setValue(100)

        self.min_segment_len = QSpinBox()
        self.min_segment_len.setRange(1, 1_000_000)
        self.min_segment_len.setValue(20)

        self.merge_tol = QSpinBox()
        self.merge_tol.setRange(0, 1_000_000)
        self.merge_tol.setValue(10)

        self.min_score = QDoubleSpinBox()
        self.min_score.setRange(-1.0, 1.0)
        self.min_score.setSingleStep(0.01)
        self.min_score.setDecimals(3)
        self.min_score.setValue(0.03)

        self.chunk_hint_label = QLabel("")
        self.chunk_hint_label.setWordWrap(True)

        self.auto_chunk_btn = QPushButton("Рассчитать параметры частей автоматически")
        self.auto_chunk_btn.clicked.connect(self._on_auto_chunk_clicked)

        self.mode_combo.currentIndexChanged.connect(lambda *_: self._toggle_chunk_fields())
        for widget in [self.chunk_size, self.overlap, self.min_segment_len, self.merge_tol, self.min_score]:
            widget.valueChanged.connect(lambda *_: self._on_chunk_param_changed())

        grid.addWidget(QLabel("n_clusters_min"), 0, 0)
        grid.addWidget(self.n_clusters_min, 0, 1)
        grid.addWidget(QLabel("n_clusters_max"), 0, 2)
        grid.addWidget(self.n_clusters_max, 0, 3)

        grid.addWidget(QLabel("k_neighbours_min"), 1, 0)
        grid.addWidget(self.k_neighbours_min, 1, 1)
        grid.addWidget(QLabel("k_neighbours_max"), 1, 2)
        grid.addWidget(self.k_neighbours_max, 1, 3)

        grid.addWidget(self.scale_cb, 2, 0, 1, 4)
        grid.addWidget(self.prefer_segments_cb, 3, 0, 1, 4)
        grid.addWidget(self.cluster_hint_label, 4, 0, 1, 4)

        grid.addWidget(QLabel("Режим"), 5, 0)
        grid.addWidget(self.mode_combo, 5, 1, 1, 3)

        grid.addWidget(QLabel("Размер части, строк"), 6, 0)
        grid.addWidget(self.chunk_size, 6, 1)
        grid.addWidget(QLabel("Перекрытие, строк"), 6, 2)
        grid.addWidget(self.overlap, 6, 3)

        grid.addWidget(QLabel("Мин. длина сегмента, строк"), 7, 0)
        grid.addWidget(self.min_segment_len, 7, 1)
        grid.addWidget(QLabel("Объединять близкие границы, строк"), 7, 2)
        grid.addWidget(self.merge_tol, 7, 3)

        grid.addWidget(QLabel("Мин. Silhouette для разбиения части"), 8, 0)
        grid.addWidget(self.min_score, 8, 1)
        grid.addWidget(self.auto_chunk_btn, 8, 2, 1, 2)

        grid.addWidget(self.chunk_hint_label, 9, 0, 1, 4)

        self._toggle_chunk_fields()
        self._update_chunk_hint()
        return group

    def _build_advanced_group(self):
        group = QGroupBox("Advanced: параметры объединения стадий")
        group.setCheckable(True)
        group.setChecked(False)
        grid = QGridLayout(group)

        self.st1_len_threshold_1 = QSpinBox()
        self.st1_len_threshold_1.setRange(0, 1_000_000)
        self.st1_len_threshold_1.setValue(0)

        self.st1_len_threshold_2 = QSpinBox()
        self.st1_len_threshold_2.setRange(0, 1_000_000)
        self.st1_len_threshold_2.setValue(20)

        self.st1_len_threshold_3 = QSpinBox()
        self.st1_len_threshold_3.setRange(0, 1_000_000)
        self.st1_len_threshold_3.setValue(40)

        self.st1_len_threshold_4 = QSpinBox()
        self.st1_len_threshold_4.setRange(0, 1_000_000)
        self.st1_len_threshold_4.setValue(60)

        self.st2_len_threshold = QSpinBox()
        self.st2_len_threshold.setRange(0, 1_000_000)
        self.st2_len_threshold.setValue(40)

        self.st1_dist_rate = QDoubleSpinBox()
        self.st1_dist_rate.setRange(0.01, 10.0)
        self.st1_dist_rate.setSingleStep(0.05)
        self.st1_dist_rate.setValue(0.3)

        self.st2_dist_rate = QDoubleSpinBox()
        self.st2_dist_rate.setRange(0.01, 10.0)
        self.st2_dist_rate.setSingleStep(0.05)
        self.st2_dist_rate.setValue(0.2)

        self.n_edge_clusters_min = QSpinBox()
        self.n_edge_clusters_min.setRange(2, 100)
        self.n_edge_clusters_min.setValue(2)

        self.n_edge_clusters_max = QSpinBox()
        self.n_edge_clusters_max.setRange(2, 100)
        self.n_edge_clusters_max.setValue(10)

        self.random_state = QSpinBox()
        self.random_state.setRange(0, 999999)
        self.random_state.setValue(42)

        self.verbose_cb = QCheckBox("verbose")
        self.verbose_cb.setChecked(False)

        st1_row = QHBoxLayout()
        for widget in [
            self.st1_len_threshold_1,
            self.st1_len_threshold_2,
            self.st1_len_threshold_3,
            self.st1_len_threshold_4,
        ]:
            st1_row.addWidget(widget)

        grid.addWidget(QLabel("st1_len_thresholds"), 0, 0)
        grid.addLayout(st1_row, 0, 1, 1, 3)

        grid.addWidget(QLabel("st2_len_threshold"), 1, 0)
        grid.addWidget(self.st2_len_threshold, 1, 1)

        grid.addWidget(QLabel("st1_dist_rate"), 2, 0)
        grid.addWidget(self.st1_dist_rate, 2, 1)
        grid.addWidget(QLabel("st2_dist_rate"), 2, 2)
        grid.addWidget(self.st2_dist_rate, 2, 3)

        grid.addWidget(QLabel("n_edge_clusters_min"), 3, 0)
        grid.addWidget(self.n_edge_clusters_min, 3, 1)
        grid.addWidget(QLabel("n_edge_clusters_max"), 3, 2)
        grid.addWidget(self.n_edge_clusters_max, 3, 3)

        grid.addWidget(QLabel("random_state"), 4, 0)
        grid.addWidget(self.random_state, 4, 1)
        grid.addWidget(self.verbose_cb, 4, 2, 1, 2)

        hint = QLabel(
            "Обычно эти параметры лучше не менять без необходимости. "
            "Thresholds и dist_rate влияют на то, насколько активно SDA объединяет короткие или похожие стадии."
        )
        hint.setWordWrap(True)
        grid.addWidget(hint, 5, 0, 1, 4)
        return group

    def _build_actions(self):
        row = QHBoxLayout()

        self.run_btn = QPushButton("Запустить SDA")
        self.cancel_btn = QPushButton("Отменить")
        self.cancel_btn.setEnabled(False)
        self.reset_btn = QPushButton("Сбросить результат")
        self.export_btn = QPushButton("Экспорт сегментов CSV")
        self.export_candidates_btn = QPushButton("Экспорт всех вариантов SDA CSV")

        self.run_btn.clicked.connect(self._run_segmentation)
        self.cancel_btn.clicked.connect(self._cancel_task)
        self.reset_btn.clicked.connect(self._reset_result)
        self.export_btn.clicked.connect(self._export_segments)
        self.export_candidates_btn.clicked.connect(self._export_candidates)

        row.addWidget(self.run_btn)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.reset_btn)
        row.addWidget(self.export_btn)
        row.addWidget(self.export_candidates_btn)
        return row

    def _build_results_group(self):
        group = QGroupBox("Результаты лучшей сегментации")
        vbox = QVBoxLayout(group)

        self.summary_label = QLabel("Результаты еще не рассчитаны.")
        self.summary_label.setWordWrap(True)
        vbox.addWidget(self.summary_label)

        self.warnings_label = QLabel("")
        self.warnings_label.setWordWrap(True)
        self.warnings_label.setStyleSheet("color: #8a5a00;")
        self.warnings_label.setVisible(False)
        vbox.addWidget(self.warnings_label)

        metrics_group = QGroupBox("Метрики и параметры выбранного варианта")
        metrics_grid = QGridLayout(metrics_group)
        self.best_metric_labels = {}

        labels = [
            ("n_segments", "Итоговых сегментов"),
            ("n_boundaries", "Границ"),
            ("candidates", "Проверено вариантов"),
            ("best_candidate_segments", "Сегментов в выбранном варианте SDA"),
            ("N_stages", "N_stages SDA"),
            ("Avg_stage_length", "Средняя длина стадии"),
            ("Avg-Silh", "Avg-Silh"),
            ("Avg-Cal-Har", "Avg-Cal-Har"),
            ("Avg-Dav-Bold", "Avg-Dav-Bold"),
            ("mode", "Режим"),
            ("chunks_processed", "Обработано частей"),
            ("chunk_count_estimate", "Оценка числа частей"),
        ]

        for i, (key, title) in enumerate(labels):
            value_label = QLabel("—")
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.best_metric_labels[key] = value_label
            row, col = divmod(i, 2)
            metrics_grid.addWidget(QLabel(f"{title}:"), row, col * 2)
            metrics_grid.addWidget(value_label, row, col * 2 + 1)

        vbox.addWidget(metrics_group)

        self.results_table = QTableView()
        self.results_table.setVisible(False)

        vbox.addWidget(QLabel("Таблица сегментов"))
        self.segments_table = QTableView()
        self.segments_table.setMinimumHeight(180)
        vbox.addWidget(self.segments_table)

        self.all_feature_plots_group = QGroupBox("Графики сегментации по выбранным признакам")
        all_plots_outer = QVBoxLayout(self.all_feature_plots_group)

        self.toggle_feature_plots_btn = QPushButton("Показать графики по всем признакам")
        self.toggle_feature_plots_btn.setCheckable(True)
        self.toggle_feature_plots_btn.setChecked(False)
        self.toggle_feature_plots_btn.clicked.connect(self._toggle_all_feature_plots)
        all_plots_outer.addWidget(self.toggle_feature_plots_btn)

        self.all_feature_plots_content = QWidget()
        self.all_feature_plots_content.setVisible(False)
        all_plots_content_layout = QVBoxLayout(self.all_feature_plots_content)
        all_plots_content_layout.setContentsMargins(0, 0, 0, 0)

        self.all_feature_plots_hint = QLabel(
            "Здесь строится отдельный график для каждого признака, использованного в SDA. "
            "На всех графиках показаны одни и те же найденные границы сегментов."
        )
        self.all_feature_plots_hint.setWordWrap(True)
        all_plots_content_layout.addWidget(self.all_feature_plots_hint)

        self.all_feature_plots_container = QWidget()
        self.all_feature_plots_layout = QVBoxLayout(self.all_feature_plots_container)
        self.all_feature_plots_layout.setContentsMargins(0, 0, 0, 0)
        self.all_feature_plots_layout.setSpacing(10)
        all_plots_content_layout.addWidget(self.all_feature_plots_container)

        all_plots_outer.addWidget(self.all_feature_plots_content)

        self.feature_canvases = []
        vbox.addWidget(self.all_feature_plots_group)

        self.length_figure = Figure(figsize=(8, 2.5), constrained_layout=True)
        self.length_canvas = ScrollFriendlyCanvas(self.length_figure)
        self.length_canvas.setMinimumHeight(220)
        vbox.addWidget(self.length_canvas)

        return group


    def _on_auto_chunk_clicked(self):
        self._auto_chunk_defaults_enabled = True
        self._apply_auto_chunk_defaults()

    def _on_chunk_param_changed(self):
        self._auto_chunk_defaults_enabled = False
        self._update_chunk_hint()

    def _estimate_chunk_count(self) -> int:
        n_rows = self._current_input_row_count()
        chunk_size = self.chunk_size.value()
        overlap = self.overlap.value()
        if n_rows <= 0 or chunk_size <= 0 or overlap >= chunk_size:
            return 0
        step = chunk_size - overlap
        return max(1, ((n_rows - 1) // step) + 1)

    def _update_chunk_hint(self):
        if not hasattr(self, "chunk_hint_label"):
            return

        n_rows = self._current_input_row_count()
        if n_rows <= 0:
            self.chunk_hint_label.setText("Нет данных для оценки числа запусков SDA.")
            return

        if self.mode_combo.currentData() == "chunked":
            chunk_size = self.chunk_size.value()
            overlap = self.overlap.value()
            step = max(1, chunk_size - overlap)
            count = self._estimate_chunk_count()
            self.chunk_hint_label.setText(
                f"В данных {n_rows} строк. SDA будет запущен примерно {count} раз(а). "
                f"Шаг между частями: {step} строк. "
                f"Автонастройка: размер части ≈ 10% данных, overlap ≈ 10% части."
            )
        else:
            self.chunk_hint_label.setText(
                f"В данных {n_rows} строк. В режиме полного набора SDA будет запущен один раз."
            )

    def _apply_auto_chunk_defaults(self):
        if not hasattr(self, "chunk_size"):
            return

        n_rows = self._current_input_row_count()
        if n_rows <= 0:
            self._update_chunk_hint()
            return

        chunk = max(100, int(round(n_rows * 0.10)))
        chunk = min(chunk, n_rows)

        overlap = max(0, int(round(chunk * 0.10)))
        overlap = min(overlap, max(0, chunk - 1))

        min_segment_len = max(3, int(round(chunk * 0.02)))
        min_segment_len = min(min_segment_len, chunk)

        merge_tol = max(1, int(round(chunk * 0.01)))
        merge_tol = min(merge_tol, chunk)

        for widget in [self.chunk_size, self.overlap, self.min_segment_len, self.merge_tol]:
            widget.blockSignals(True)

        self.chunk_size.setMaximum(max(100, n_rows))
        self.overlap.setMaximum(max(0, n_rows - 1))
        self.min_segment_len.setMaximum(max(1, n_rows))
        self.merge_tol.setMaximum(max(0, n_rows))

        self.chunk_size.setValue(chunk)
        self.overlap.setValue(overlap)
        self.min_segment_len.setValue(min_segment_len)
        self.merge_tol.setValue(merge_tol)

        for widget in [self.chunk_size, self.overlap, self.min_segment_len, self.merge_tol]:
            widget.blockSignals(False)

        self._update_chunk_hint()

    def _toggle_chunk_fields(self):
        is_chunked = self.mode_combo.currentData() == "chunked"
        for w in [
            self.chunk_size,
            self.overlap,
            self.min_segment_len,
            self.merge_tol,
            self.min_score,
            self.auto_chunk_btn,
        ]:
            w.setEnabled(is_chunked)
        self._update_chunk_hint()


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
        st1_thresholds = sorted(set([
            self.st1_len_threshold_1.value(),
            self.st1_len_threshold_2.value(),
            self.st1_len_threshold_3.value(),
            self.st1_len_threshold_4.value(),
        ]))

        n_clusters_min = self.n_clusters_min.value()
        n_clusters_max = self.n_clusters_max.value()
        k_neighbours_min = self.k_neighbours_min.value()
        k_neighbours_max = self.k_neighbours_max.value()

        return {
            "n_clusters_min": n_clusters_min,
            "n_clusters_max": n_clusters_max,
            "k_neighbours_min": k_neighbours_min,
            "k_neighbours_max": k_neighbours_max,
            "st1_len_thresholds": st1_thresholds,
            "st2_len_thresholds": [self.st2_len_threshold.value()],
            "st1_dist_rate": self.st1_dist_rate.value(),
            "st2_dist_rate": self.st2_dist_rate.value(),


            "n_cl_max_thr": [n_clusters_max],
            "k_neighb_max_thr": [k_neighbours_max],

            "n_edge_clusters_min": self.n_edge_clusters_min.value(),
            "n_edge_clusters_max": self.n_edge_clusters_max.value(),
            "random_state": self.random_state.value(),
            "scale": self.scale_cb.isChecked(),
            "verbose": self.verbose_cb.isChecked(),

            # Выбор лучшего результата.
            "prefer_min_segments": self.prefer_segments_cb.isChecked(),
            "preferred_min_segments": n_clusters_min,

            # Chunked mode.
            "mode": self.mode_combo.currentData(),
            "chunk_size": self.chunk_size.value(),
            "overlap": self.overlap.value(),
            "min_segment_len": self.min_segment_len.value(),
            "merge_boundaries_tolerance": self.merge_tol.value(),
            "min_score_to_split": self.min_score.value(),
            "chunk_count_estimate": self._estimate_chunk_count(),
        }


    def _run_segmentation(self):
        selected_columns = [cb.text() for cb in self.column_checkboxes if cb.isChecked()]
        if not selected_columns:
            self._show_error("Выберите хотя бы один признак.")
            return

        if self.n_clusters_min.value() > self.n_clusters_max.value():
            self._show_error("n_clusters_min не может быть больше n_clusters_max.")
            return
        if self.k_neighbours_min.value() > self.k_neighbours_max.value():
            self._show_error("k_neighbours_min не может быть больше k_neighbours_max.")
            return
        if self.mode_combo.currentData() == "chunked" and self.overlap.value() >= self.chunk_size.value():
            self._show_error("Перекрытие должно быть меньше размера части.")
            return

        params = self._collect_params()
        request = self.vm.build_segmentation_request(
            selected_columns,
            params,
            source_key=self._selected_input_key,
            output_name=self._output_name,
        )

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


    @staticmethod
    def _format_metric_value(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    def _show_result(self, result):
        self.current_result = result

        if "Avg-Silh" in result.results_table.columns:
            sorted_results = result.results_table.sort_values(by="Avg-Silh", ascending=False)
        else:
            sorted_results = result.results_table
        self.results_table.setModel(DataFrameModel(sorted_results.reset_index(drop=True)))

        self.segments_table.setModel(DataFrameModel(result.segments_table.reset_index(drop=True).fillna("")))

        for key, label in self.best_metric_labels.items():
            label.setText(self._format_metric_value(result.summary.get(key)))

        warnings = result.summary.get("warnings", [])
        if warnings:
            self.warnings_label.setText("Предупреждения:\n" + "\n".join(f"• {w}" for w in warnings))
            self.warnings_label.setVisible(True)
        else:
            self.warnings_label.setVisible(False)

        self.summary_label.setText(
            "Выбран лучший вариант SDA по метрикам качества и ограничению на слишком простую сегментацию. "
            "Если итоговых сегментов меньше, чем n_clusters_min, это означает, что SDA после первичного поиска "
            "объединил близкие стадии или не нашел устойчивого более детального разбиения."
        )

        if self.toggle_feature_plots_btn.isChecked():
            self._draw_all_feature_plots(result)
        else:
            self._clear_all_feature_plots()
        self._draw_segment_lengths(result)

    def _toggle_all_feature_plots(self, checked: bool):
        self.all_feature_plots_content.setVisible(checked)
        self.toggle_feature_plots_btn.setText(
            "Скрыть графики по всем признакам" if checked else "Показать графики по всем признакам"
        )

        if checked and self.current_result is not None:
            self._draw_all_feature_plots(self.current_result)
        else:
            self._clear_all_feature_plots()

    def _clear_all_feature_plots(self):
        if not hasattr(self, "all_feature_plots_layout"):
            return

        while self.all_feature_plots_layout.count():
            item = self.all_feature_plots_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.feature_canvases = []

    def _draw_all_feature_plots(self, result):
        if not hasattr(self, "all_feature_plots_group"):
            return

        self._clear_all_feature_plots()

        if result is None or result.segmented_data is None or result.segmented_data.empty:
            return

        selected_columns = [
            col for col in result.selected_columns
            if col in result.segmented_data.columns
        ]

        if not selected_columns:
            label = QLabel("Нет доступных признаков для построения графиков.")
            self.all_feature_plots_layout.addWidget(label)
            return

        max_plots = 30
        visible_columns = selected_columns[:max_plots]

        if len(selected_columns) > max_plots:
            label = QLabel(
                f"Выбрано {len(selected_columns)} признаков. Для скорости показаны первые {max_plots}. "
                "При необходимости можно выбрать меньше признаков и перезапустить сегментацию."
            )
            label.setWordWrap(True)
            self.all_feature_plots_layout.addWidget(label)

        x = list(range(len(result.segmented_data)))

        for col in visible_columns:
            title = QLabel(col)
            title.setStyleSheet("font-weight: bold;")
            self.all_feature_plots_layout.addWidget(title)

            fig = Figure(figsize=(8, 2.4), constrained_layout=True)
            canvas = ScrollFriendlyCanvas(fig)
            canvas.setMinimumHeight(220)

            ax = fig.add_subplot(111)
            y = result.segmented_data[col].values
            ax.plot(x, y, linewidth=1.0, label=col)

            for edge in result.edges:
                ax.axvline(edge, color="red", linestyle="--", alpha=0.55)

            ax.set_title(f"Сегментация по признаку: {col}")
            ax.set_xlabel("Индекс")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right")

            self.all_feature_plots_layout.addWidget(canvas)
            self.feature_canvases.append(canvas)

            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            self.all_feature_plots_layout.addWidget(line)

        for canvas in self.feature_canvases:
            canvas.draw_idle()

    def _draw_segment_lengths(self, result):
        self.length_figure.clear()
        ax = self.length_figure.add_subplot(111)

        if result.segments_table is None or result.segments_table.empty or "length" not in result.segments_table.columns:
            ax.set_title("Длины сегментов")
            ax.text(0.5, 0.5, "Нет данных", ha="center", va="center")
            self.length_canvas.draw_idle()
            return

        lengths = result.segments_table["length"].astype(float).values
        ax.bar(range(len(lengths)), lengths)
        ax.set_title("Длины сегментов")
        ax.set_xlabel("segment_id")
        ax.set_ylabel("length")
        ax.grid(alpha=0.3)
        self.length_canvas.draw_idle()

    def _reset_result(self):
        self.vm.reset_result()
        self.current_result = None
        self.results_table.setModel(None)
        self.segments_table.setModel(None)

        for label in getattr(self, "best_metric_labels", {}).values():
            label.setText("—")

        self.warnings_label.setVisible(False)
        self.summary_label.setText("Результаты еще не рассчитаны.")
        self.length_figure.clear()
        self.length_canvas.draw_idle()
        self._clear_all_feature_plots()
        if hasattr(self, "toggle_feature_plots_btn"):
            self.toggle_feature_plots_btn.setChecked(False)
            self.toggle_feature_plots_btn.setText("Показать графики по всем признакам")
        if hasattr(self, "all_feature_plots_content"):
            self.all_feature_plots_content.setVisible(False)

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

    def _export_candidates(self):
        if self.current_result is None:
            self._show_error("Нет результатов SDA для экспорта.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт всех вариантов SDA",
            str(Path.home() / "sda_candidates.csv"),
            "CSV Files (*.csv)",
        )
        if not file_path:
            return

        self.current_result.results_table.to_csv(file_path, index=False)
        self._show_info(f"Экспортировано: {file_path}")


    def _show_error(self, message: str):
        QMessageBox.critical(self, "Ошибка", message)

    def _show_info(self, message: str):
        self.status_label.setText(f"Статус: {message}")
