from PySide6.QtWidgets import (
    QLabel, QVBoxLayout, QHBoxLayout, QGroupBox,
    QCheckBox, QComboBox, QPushButton, QSpinBox,
    QScrollArea, QWidget
)

from matplotlib.figure import Figure
from ui.canvas import ScrollFriendlyCanvas
from ui.pages.base_page import BasePage
from services.feature_service import FeatureService
from PySide6.QtWidgets import QTableView
from PySide6.QtCore import QAbstractTableModel, Qt
from ui.models.dataframe_model import DataFrameModel


class FeaturesPage(BasePage):

    def __init__(self, data_vm):
        super().__init__()

        self.vm = data_vm
        self.df = None

        self.vm.data_loaded.connect(self.on_data_loaded)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setSpacing(14)

        # ---------------- TITLE ----------------

        title = QLabel("Формирование признаков")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # ---------------- COLUMN SELECTOR ----------------

        layout.addWidget(QLabel("Выберите столбцы:"))

        self.columns_container = QVBoxLayout()
        self.column_checkboxes = []

        columns_widget = QWidget()
        columns_widget.setLayout(self.columns_container)

        layout.addWidget(columns_widget)

        # ---------------- WINDOW PARAMETERS ----------------

        layout.addWidget(self._window_group())

        # ---------------- FEATURE GROUPS ----------------

        layout.addWidget(self._stat_features())
        layout.addWidget(self._dynamic_features())
        layout.addWidget(self._energy_features())

        # ---------------- PLOT ----------------

        self.figure = Figure(figsize=(5, 3))
        self.canvas = ScrollFriendlyCanvas(self.figure)
        self.canvas.setMinimumHeight(300)

        figure_container = QWidget()
        figure_layout = QVBoxLayout(figure_container)
        figure_layout.setContentsMargins(0, 0, 0, 0)
        figure_layout.addWidget(self.canvas)

        layout.addWidget(figure_container)

        self.table = QTableView()
        layout.addWidget(QLabel("Сгенерированные признаки"))
        layout.addWidget(self.table)

        # ---------------- GENERATE BUTTON ----------------

        self.generate_btn = QPushButton("Сгенерировать признаки")
        self.generate_btn.clicked.connect(self.generate_features)

        layout.addWidget(self.generate_btn)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(scroll)

    # ============================================================
    # WINDOW PARAMETERS
    # ============================================================

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

    # ============================================================
    # STAT FEATURES
    # ============================================================

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

        layout.addWidget(self.mean_cb)
        layout.addWidget(self.std_cb)
        layout.addWidget(self.var_cb)
        layout.addWidget(self.min_cb)
        layout.addWidget(self.max_cb)
        layout.addWidget(self.skew_cb)
        layout.addWidget(self.kurt_cb)

        group.setLayout(layout)

        return group

    # ============================================================
    # DYNAMIC FEATURES
    # ============================================================

    def _dynamic_features(self):

        group = QGroupBox("Динамические признаки")
        layout = QVBoxLayout()

        self.diff_cb = QCheckBox("First difference")
        self.gradient_cb = QCheckBox("Gradient")
        self.roc_cb = QCheckBox("Rate of change")

        layout.addWidget(self.diff_cb)
        layout.addWidget(self.gradient_cb)
        layout.addWidget(self.roc_cb)

        group.setLayout(layout)

        return group

    # ============================================================
    # ENERGY FEATURES
    # ============================================================

    def _energy_features(self):

        group = QGroupBox("Амплитудные / энергетические признаки")
        layout = QVBoxLayout()

        self.rms_cb = QCheckBox("RMS")
        self.energy_cb = QCheckBox("Signal energy")
        self.ptp_cb = QCheckBox("Peak-to-peak")

        layout.addWidget(self.rms_cb)
        layout.addWidget(self.energy_cb)
        layout.addWidget(self.ptp_cb)

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

    def generate_features(self):

        if self.df is None:
            return

        selected_columns = self.get_selected_columns()

        if not selected_columns:
            print("Не выбраны столбцы")
            return

        window = self.window_size.value()
        step = self.step_size.value()

        selected_features = []

        if self.mean_cb.isChecked():
            selected_features.append("mean")
        if self.std_cb.isChecked():
            selected_features.append("std")
        if self.var_cb.isChecked():
            selected_features.append("var")
        if self.min_cb.isChecked():
            selected_features.append("min")
        if self.max_cb.isChecked():
            selected_features.append("max")
        if self.skew_cb.isChecked():
            selected_features.append("skew")
        if self.kurt_cb.isChecked():
            selected_features.append("kurt")
        if self.diff_cb.isChecked():
            selected_features.append("diff")
        if self.gradient_cb.isChecked():
            selected_features.append("gradient")
        if self.roc_cb.isChecked():
            selected_features.append("roc")
        if self.rms_cb.isChecked():
            selected_features.append("rms")
        if self.energy_cb.isChecked():
            selected_features.append("energy")
        if self.ptp_cb.isChecked():
            selected_features.append("ptp")

        import pandas as pd

        all_features = []

        for col in selected_columns:
            series = self.df[col]

            features_df = FeatureService.extract_features(
                series,
                window,
                step,
                selected_features
            )

            # 👉 переименовываем колонки
            features_df = features_df.add_prefix(f"{col}_")

            all_features.append(features_df)

        # 👉 объединяем по колонкам
        self.features_df = pd.concat(all_features, axis=1)

        model = DataFrameModel(self.features_df)
        self.table.setModel(model)

        self.update_feature_plot()
        self.update_correlation_heatmap()

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
        ax.set_xticklabels(corr.columns, rotation=45)

        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns)

        ax.set_title("Feature correlation")

        self.figure.colorbar(im)

        self.canvas.draw()