from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import (
    QLabel, QVBoxLayout, QGroupBox,
    QCheckBox, QPushButton, QSpinBox,
    QScrollArea, QWidget, QTableView, QMessageBox, QProgressBar, QHBoxLayout
)
from matplotlib.figure import Figure

from ui.canvas import ScrollFriendlyCanvas
from ui.pages.base_page import BasePage
from services.feature_service import FeatureService
from ui.models.dataframe_model import DataFrameModel
from workers.pipeline_worker import PipelineWorker


class FeaturesPage(BasePage):

    def __init__(self, data_vm):
        super().__init__()

        self.vm = data_vm
        self.df = None

        self.features_df = None
        self.thread_pool = QThreadPool.globalInstance()
        self.current_worker = None

        self.vm.data_loaded.connect(self.on_data_loaded)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setSpacing(14)

        title = QLabel("Формирование признаков")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        layout.addWidget(QLabel("Выберите столбцы:"))

        self.columns_container = QVBoxLayout()
        self.column_checkboxes = []

        columns_widget = QWidget()
        columns_widget.setLayout(self.columns_container)

        layout.addWidget(columns_widget)

        layout.addWidget(self._window_group())

        layout.addWidget(self._stat_features())
        layout.addWidget(self._dynamic_features())
        layout.addWidget(self._energy_features())

        self.figure = Figure(figsize=(8, 5), constrained_layout=True)
        self.canvas = ScrollFriendlyCanvas(self.figure)
        self.canvas.setMinimumHeight(300)

        figure_container = QWidget()
        figure_layout = QVBoxLayout(figure_container)
        figure_layout.setContentsMargins(0, 0, 0, 0)
        figure_layout.addWidget(self.canvas)

        layout.addWidget(figure_container)

        self.table = QTableView()
        self.table.setMinimumHeight(220)
        layout.addWidget(QLabel("Сгенерированные признаки"))
        layout.addWidget(self.table)

        self.status_label = QLabel("Статус: ожидание")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)

        action_row = QHBoxLayout()
        self.generate_btn = QPushButton("Сгенерировать признаки")
        self.generate_btn.clicked.connect(self.generate_features)
        self.cancel_btn = QPushButton("Отменить")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_generation)
        action_row.addWidget(self.generate_btn)
        action_row.addWidget(self.cancel_btn)
        layout.addLayout(action_row)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(scroll)

    def _window_group(self):

        group = QGroupBox("Sliding window параметры")
        layout = QVBoxLayout()

        self.window_size = QSpinBox()
        self.window_size.setRange(2, 10000)
        self.window_size.setValue(50)

        self.step_size = QSpinBox()
        self.step_size.setRange(1, 10000)
        self.step_size.setValue(10)

        layout.addWidget(QLabel("Размер окна"))
        layout.addWidget(self.window_size)

        layout.addWidget(QLabel("Шаг окна"))
        layout.addWidget(self.step_size)

        group.setLayout(layout)

        return group


    def _stat_features(self):

        group = QGroupBox("Статистические признаки")
        layout = QVBoxLayout()

        self.mean_cb = QCheckBox("Mean")
        self.std_cb = QCheckBox("Std")
        self.var_cb = QCheckBox("Variance")
        self.min_cb = QCheckBox("Min")
        self.max_cb = QCheckBox("Max")
        self.skew_cb = QCheckBox("Skewness")
        self.kurt_cb = QCheckBox("Kurtosis")

        for cb in [self.mean_cb, self.std_cb, self.var_cb, self.min_cb, self.max_cb, self.skew_cb, self.kurt_cb]:
            layout.addWidget(cb)

        group.setLayout(layout)

        return group


    def _dynamic_features(self):

        group = QGroupBox("Динамические признаки")
        layout = QVBoxLayout()

        self.diff_cb = QCheckBox("First difference")
        self.gradient_cb = QCheckBox("Gradient")
        self.roc_cb = QCheckBox("Rate of change")

        for cb in [self.diff_cb, self.gradient_cb, self.roc_cb]:
            layout.addWidget(cb)

        group.setLayout(layout)
        return group


    def _energy_features(self):

        group = QGroupBox("Амплитудные / энергетические признаки")
        layout = QVBoxLayout()

        self.rms_cb = QCheckBox("RMS")
        self.energy_cb = QCheckBox("Signal energy")
        self.ptp_cb = QCheckBox("Peak-to-peak")

        for cb in [self.rms_cb, self.energy_cb, self.ptp_cb]:
            layout.addWidget(cb)

        group.setLayout(layout)
        return group


    def on_data_loaded(self, df):

        self.df = df

        # очистка старых чекбоксов
        for cb in self.column_checkboxes:
            self.columns_container.removeWidget(cb)
            cb.deleteLater()

        self.column_checkboxes = []

        numeric_columns = df.select_dtypes(include="number").columns

        for col in numeric_columns:
            cb = QCheckBox(col)
            self.columns_container.addWidget(cb)
            self.column_checkboxes.append(cb)

    def get_selected_columns(self):
        return [cb.text() for cb in self.column_checkboxes if cb.isChecked()]

    def cancel_generation(self):
        if self.current_worker is not None:
            self.current_worker.cancel()
            self.status_label.setText("Статус: отмена...")

    def generate_features(self):
        if self.df is None:
            return

        selected_columns = self.get_selected_columns()
        if not selected_columns:
            QMessageBox.warning(self, "Ошибка", "Не выбраны столбцы")
            return

        selected_features = self._get_selected_features()
        if not selected_features:
            QMessageBox.warning(self, "Ошибка", "Выберите хотя бы один признак")
            return

        window = self.window_size.value()
        step = self.step_size.value()

        self._set_busy(True)
        self.status_label.setText("Статус: запуск вычисления признаков")
        self.progress_bar.setValue(0)

        self.current_worker = PipelineWorker(
            self._compute_features,
            self.df.copy(),
            selected_columns,
            window,
            step,
            selected_features,
        )
        self.current_worker.signals.progress.connect(self._on_progress)
        self.current_worker.signals.result.connect(self._on_result)
        self.current_worker.signals.error.connect(self._on_error)
        self.current_worker.signals.finished.connect(self._on_finished)
        self.thread_pool.start(self.current_worker)

    def _compute_features(self, df, selected_columns, window, step, selected_features, progress_callback=None, is_cancelled=None):
        import pandas as pd

        all_features = []
        total_cols = max(1, len(selected_columns))
        for idx, col in enumerate(selected_columns, start=1):
            if is_cancelled and is_cancelled():
                raise RuntimeError("Задача отменена пользователем.")

            features_df = FeatureService.extract_features(
                df[col], window, step, selected_features, progress_callback=progress_callback, is_cancelled=is_cancelled
            ).add_prefix(f"{col}_")
            all_features.append(features_df)

            if progress_callback:
                progress_callback.emit(int((idx / total_cols) * 100), f"Обработана колонка: {col}")

        result_df = pd.concat(all_features, axis=1)
        return {
            "features_df": result_df,
            "params": {
                "window_size": window,
                "step_size": step,
                "selected_columns": selected_columns,
                "selected_features": selected_features,
            },
        }

    def _on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(f"Статус: {message}")

    def _on_result(self, payload):
        self.features_df = payload["features_df"]
        self.vm.project.set_features(self.features_df, params=payload["params"])
        self.table.setModel(DataFrameModel(self.features_df))
        self.update_feature_plot()
        self.update_correlation_heatmap()

    def _on_error(self, message):
        if "отменена" in message.lower():
            self.status_label.setText("Статус: задача отменена")
            return
        QMessageBox.critical(self, "Ошибка", message)

    def _on_finished(self, cancelled):
        self._set_busy(False)
        if cancelled:
            self.status_label.setText("Статус: задача отменена")
        elif self.progress_bar.value() < 100:
            self.progress_bar.setValue(100)
            self.status_label.setText("Статус: выполнено")
        self.current_worker = None

    def _set_busy(self, busy: bool):
        self.generate_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)

    def _get_selected_features(self):
        selected_features = []
        if self.mean_cb.isChecked(): selected_features.append("mean")
        if self.std_cb.isChecked(): selected_features.append("std")
        if self.var_cb.isChecked(): selected_features.append("var")
        if self.min_cb.isChecked(): selected_features.append("min")
        if self.max_cb.isChecked(): selected_features.append("max")
        if self.skew_cb.isChecked(): selected_features.append("skew")
        if self.kurt_cb.isChecked(): selected_features.append("kurt")
        if self.diff_cb.isChecked(): selected_features.append("diff")
        if self.gradient_cb.isChecked(): selected_features.append("gradient")
        if self.roc_cb.isChecked(): selected_features.append("roc")
        if self.rms_cb.isChecked(): selected_features.append("rms")
        if self.energy_cb.isChecked(): selected_features.append("energy")
        if self.ptp_cb.isChecked(): selected_features.append("ptp")
        return selected_features

    def update_feature_plot(self):

        if self.df is None:
            return

        selected_columns = self.get_selected_columns()
        if not selected_columns:
            return

        column = selected_columns[0]

        values = self.df[column].values

        window = self.window_size.value()
        step = self.step_size.value()

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.plot(values, label="Time series")

        for start in range(0, len(values) - window + 1, step):
            ax.axvspan(start, start + window, alpha=0.1)

        ax.set_title("Sliding windows")
        ax.legend()

        self.canvas.draw()

    def update_correlation_heatmap(self):

        if self.features_df is None:
            return

        corr = self.features_df.corr()

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        im = ax.imshow(corr, aspect="auto")

        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")

        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)

        ax.set_title("Feature correlation")

        self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self.figure.tight_layout()

        self.canvas.draw()
