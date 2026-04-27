from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
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
)

from ui.canvas import ScrollFriendlyCanvas
from ui.models.dataframe_model import DataFrameModel
from ui.pages.base_page import BasePage
from viewmodels.markov_vm import MarkovViewModel
from workers.pipeline_worker import PipelineWorker


class MarkovPage(BasePage):
    def __init__(self, data_vm):
        super().__init__()
        self.vm = MarkovViewModel(data_vm.project)
        self.current_result = None

        self.thread_pool = QThreadPool.globalInstance()
        self.current_worker = None

        self.vm.source_info_ready.connect(self._render_source_info)
        self.vm.model_ready.connect(self._show_result)
        self.vm.model_reset.connect(self._clear_results)
        self.vm.error_occurred.connect(self._show_error)
        self.vm.info_changed.connect(self._show_info)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setSpacing(12)

        title = QLabel("Markov modeling")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        layout.addWidget(self._build_source_group())
        layout.addWidget(self._build_params_group())
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
        self.vm.refresh_source_info()

    def _build_source_group(self):
        group = QGroupBox("Источник данных")
        grid = QGridLayout(group)

        self.source_label = QLabel("Источник: clusters")
        self.source_status = QLabel("Статус: ожидание")
        self.sequence_len_label = QLabel("Длина последовательности: 0")
        self.unique_states_label = QLabel("Уникальных состояний: 0")

        grid.addWidget(self.source_label, 0, 0)
        grid.addWidget(self.source_status, 1, 0, 1, 2)
        grid.addWidget(self.sequence_len_label, 2, 0)
        grid.addWidget(self.unique_states_label, 2, 1)
        return group

    def _build_params_group(self):
        group = QGroupBox("Параметры")
        grid = QGridLayout(group)

        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 10)
        self.order_spin.setValue(1)

        self.min_frequency_spin = QSpinBox()
        self.min_frequency_spin.setRange(1, 1000)
        self.min_frequency_spin.setValue(1)

        self.normalize_cb = QCheckBox("Нормализовать в вероятности")
        self.normalize_cb.setChecked(True)

        self.sequential_cb = QCheckBox("Учитывать только последовательные переходы")
        self.sequential_cb.setChecked(True)

        grid.addWidget(QLabel("Порядок цепи"), 0, 0)
        grid.addWidget(self.order_spin, 0, 1)
        grid.addWidget(QLabel("Минимальная частота перехода"), 1, 0)
        grid.addWidget(self.min_frequency_spin, 1, 1)
        grid.addWidget(self.normalize_cb, 2, 0, 1, 2)
        grid.addWidget(self.sequential_cb, 3, 0, 1, 2)

        return group

    def _build_actions(self):
        row = QHBoxLayout()

        self.run_btn = QPushButton("Построить модель")
        self.cancel_btn = QPushButton("Отменить")
        self.cancel_btn.setEnabled(False)
        self.reset_btn = QPushButton("Сбросить результат")
        self.export_btn = QPushButton("Экспорт матрицы CSV")

        self.run_btn.clicked.connect(self._run_model)
        self.cancel_btn.clicked.connect(self._cancel_task)
        self.reset_btn.clicked.connect(self.vm.reset_result)
        self.export_btn.clicked.connect(self._export_csv)

        row.addWidget(self.run_btn)
        row.addWidget(self.cancel_btn)
        row.addWidget(self.reset_btn)
        row.addWidget(self.export_btn)
        return row

    def _build_results_group(self):
        group = QGroupBox("Результаты")
        layout = QVBoxLayout(group)

        self.summary_label = QLabel("Результаты еще не рассчитаны.")
        layout.addWidget(self.summary_label)

        layout.addWidget(QLabel("Transition counts"))
        self.counts_table = QTableView()
        self.counts_table.setMinimumHeight(180)
        layout.addWidget(self.counts_table)

        layout.addWidget(QLabel("Transition probabilities"))
        self.prob_table = QTableView()
        self.prob_table.setMinimumHeight(180)
        layout.addWidget(self.prob_table)

        layout.addWidget(QLabel("Transitions (long-form)"))
        self.long_table = QTableView()
        self.long_table.setMinimumHeight(180)
        layout.addWidget(self.long_table)

        self.figure = Figure(figsize=(8, 4), constrained_layout=True)
        self.canvas = ScrollFriendlyCanvas(self.figure)
        self.canvas.setMinimumHeight(280)
        layout.addWidget(self.canvas)

        return group

    def _run_model(self):
        request = self.vm.build_model_request(
            order=self.order_spin.value(),
            normalize=self.normalize_cb.isChecked(),
            sequential_only=self.sequential_cb.isChecked(),
            min_frequency=self.min_frequency_spin.value(),
        )
        self._set_busy(True)
        self.status_label.setText("Статус: выполняется построение модели")
        self.progress_bar.setValue(0)

        self.current_worker = PipelineWorker(self.vm.execute_model, **request)
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
        self.vm.apply_model_result(result)

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
        self.counts_table.setModel(DataFrameModel(result.transition_counts.reset_index().rename(columns={"index": "history_state"})))
        self.prob_table.setModel(DataFrameModel(result.transition_probabilities.reset_index().rename(columns={"index": "history_state"})))
        self.long_table.setModel(DataFrameModel(result.transitions_long_table))

        summary = result.summary
        self.summary_label.setText(
            " | ".join(
                [
                    f"order={summary.get('order')}",
                    f"sequence={summary.get('sequence_length')}",
                    f"states={summary.get('unique_state_count')}",
                    f"transitions={summary.get('observed_transitions')}",
                    f"sparsity={summary.get('sparsity', 0):.3f}",
                    f"entropy={summary.get('weighted_entropy', 0):.3f}",
                ]
            )
        )
        self._draw_heatmap(result.transition_probabilities)

    def _draw_heatmap(self, probabilities_df):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if probabilities_df.empty:
            ax.text(0.5, 0.5, "Нет переходов для отображения", ha="center", va="center")
            ax.axis("off")
            self.canvas.draw_idle()
            return

        display_df = probabilities_df
        max_rows = 25
        max_cols = 25
        if len(display_df.index) > max_rows:
            display_df = display_df.iloc[:max_rows, :]
        if len(display_df.columns) > max_cols:
            display_df = display_df.iloc[:, :max_cols]

        matrix = display_df.to_numpy(dtype=float)
        img = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=max(float(np.max(matrix)), 1e-9))
        ax.set_title("Heatmap переходных вероятностей")
        ax.set_xlabel("Next state")
        ax.set_ylabel("History state")

        ax.set_xticks(range(len(display_df.columns)))
        ax.set_xticklabels([str(c) for c in display_df.columns], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(display_df.index)))
        ax.set_yticklabels([str(i) for i in display_df.index], fontsize=7)

        self.figure.colorbar(img, ax=ax, fraction=0.03, pad=0.02)
        self.canvas.draw_idle()

    def _export_csv(self):
        if self.current_result is None:
            self._show_error("Нет результата для экспорта")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт матрицы вероятностей",
            str(Path.home() / "markov_transition_probabilities.csv"),
            "CSV Files (*.csv)",
        )
        if not file_path:
            return

        try:
            self.vm.export_probabilities_csv(file_path)
            self._show_info(f"Экспортировано: {file_path}")
        except Exception as exc:
            self._show_error(str(exc))

    def _render_source_info(self, info: dict):
        self.source_label.setText(f"Источник: {info.get('source_name', 'clusters')}")
        self.source_status.setText(f"Статус: {info.get('message', '')}")
        self.sequence_len_label.setText(f"Длина последовательности: {info.get('sequence_length', 0)}")
        self.unique_states_label.setText(f"Уникальных состояний: {info.get('unique_states', 0)}")

    def _clear_results(self):
        self.current_result = None
        self.summary_label.setText("Результаты еще не рассчитаны")
        self.counts_table.setModel(None)
        self.prob_table.setModel(None)
        self.long_table.setModel(None)
        self.figure.clear()
        self.canvas.draw_idle()

    def _show_error(self, message: str):
        QMessageBox.critical(self, "Ошибка", message)

    def _show_info(self, message: str):
        QMessageBox.information(self, "Информация", message)
