from PySide6.QtWidgets import (
    QLabel, QVBoxLayout, QGroupBox, QRadioButton,
    QComboBox, QPushButton, QSpinBox, QDoubleSpinBox, QScrollArea, QWidget
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ui.canvas import ScrollFriendlyCanvas
from ui.pages.base_page import BasePage
from services.preprocessing_service import PreprocessingService
from viewmodels.data_vm import DataViewModel


class PreprocessingPage(BasePage):
    def __init__(self, project):
        super().__init__()

        self.project = project
        self.df = None

        self.vm = project.data_vm
        self.vm.data_loaded.connect(self.on_data_loaded)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setSpacing(14)

        title = QLabel("Предварительная обработка данных")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.column_selector = QComboBox()
        self.column_selector.setEnabled(False)

        layout.addWidget(QLabel("Анализируемый столбец:"))
        layout.addWidget(self.column_selector)

        #масштабирование
        self.scale_group = self._scaling_group()
        layout.addWidget(self.scale_group)

        #пропуски
        self.missing_group = self._missing_group()
        layout.addWidget(self.missing_group)

        #сглаживание
        self.smoothing_group = self._smoothing_group()
        layout.addWidget(self.smoothing_group)

        self.figure = Figure(figsize=(5, 3))
        self.canvas = ScrollFriendlyCanvas(self.figure)
        self.canvas.setMinimumHeight(300)

        figure_container = QWidget()
        figure_layout = QVBoxLayout(figure_container)
        figure_layout.setContentsMargins(0, 0, 0, 0)
        figure_layout.addWidget(self.canvas)

        layout.addWidget(figure_container)

        self.apply_btn = QPushButton("Применить изменения")
        self.apply_btn.clicked.connect(self.apply)
        layout.addWidget(self.apply_btn)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(scroll)

        self.update_preview()

    def _scaling_group(self):
        box = QGroupBox("Масштабирование")
        layout = QVBoxLayout(box)

        self.scale_none = QRadioButton("Не применять")
        self.scale_minmax = QRadioButton("Min–Max")
        self.scale_z = QRadioButton("Z-score")
        self.scale_robust = QRadioButton("Robust scaling")

        self.scale_none.setChecked(True)

        for btn in [
            self.scale_none,
            self.scale_minmax,
            self.scale_z,
            self.scale_robust
        ]:
            btn.toggled.connect(self.update_preview)
            layout.addWidget(btn)

        return box

    def _missing_group(self):
        box = QGroupBox("Обработка пропусков")
        layout = QVBoxLayout(box)

        self.miss_none = QRadioButton("Не обрабатывать")
        self.miss_drop = QRadioButton("Удалить строки")
        self.miss_mean = QRadioButton("Заполнить mean")
        self.miss_median = QRadioButton("Заполнить median")
        self.miss_interp = QRadioButton("Интерполяция")

        self.miss_none.setChecked(True)

        for btn in [
            self.miss_none,
            self.miss_drop,
            self.miss_mean,
            self.miss_median,
            self.miss_interp
        ]:
            btn.toggled.connect(self.update_preview)
            layout.addWidget(btn)

        return box

    def _smoothing_group(self):
        box = QGroupBox("Сглаживание")
        layout = QVBoxLayout(box)

        self.smooth_none = QRadioButton("Не применять")
        self.smooth_ma = QRadioButton("Скользящее среднее")
        self.smooth_median = QRadioButton("Медианный фильтр")
        self.smooth_ewm = QRadioButton("Экспоненциальное сглаживание")

        self.smooth_none.setChecked(True)

        self.window_spin = QSpinBox()
        self.window_spin.setRange(2, 100)
        self.window_spin.setValue(5)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.01, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(0.3)

        for btn in [
            self.smooth_none,
            self.smooth_ma,
            self.smooth_median,
            self.smooth_ewm
        ]:
            btn.toggled.connect(self.update_preview)
            layout.addWidget(btn)

        layout.addWidget(QLabel("Окно сглаживания:"))
        layout.addWidget(self.window_spin)

        layout.addWidget(QLabel("Alpha (для EWM):"))
        layout.addWidget(self.alpha_spin)

        return box

    def update_preview(self):
        if self.df is None:
            return

        col = self.column_selector.currentText()
        series = self.df[col]

        # пропуски
        if self.miss_drop.isChecked():
            series = PreprocessingService.handle_missing(series, "drop")
        elif self.miss_mean.isChecked():
            series = PreprocessingService.handle_missing(series, "mean")
        elif self.miss_median.isChecked():
            series = PreprocessingService.handle_missing(series, "median")
        elif self.miss_interp.isChecked():
            series = PreprocessingService.handle_missing(series, "interpolate")

        # сглаживание
        if self.smooth_ma.isChecked():
            series = PreprocessingService.smooth(
                series, "moving_average", window=self.window_spin.value()
            )
        elif self.smooth_median.isChecked():
            series = PreprocessingService.smooth(
                series, "median", window=self.window_spin.value()
            )
        elif self.smooth_ewm.isChecked():
            series = PreprocessingService.smooth(
                series, "ewm", alpha=self.alpha_spin.value()
            )

        # масштабирование
        if self.scale_minmax.isChecked():
            series = PreprocessingService.scale(series, "minmax")
        elif self.scale_z.isChecked():
            series = PreprocessingService.scale(series, "zscore")
        elif self.scale_robust.isChecked():
            series = PreprocessingService.scale(series, "robust")

        # график
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(series.values)
        ax.set_title("Предпросмотр предобработки")

        self.canvas.draw()

    def apply(self):
        # все потом....
        pass

    def on_data_loaded(self, df):
        self.df = df
        print("DATA LOADED", df.shape)
        numeric_cols = df.select_dtypes(include="number").columns
        self.column_selector.clear()
        self.column_selector.addItems(numeric_cols)
        self.column_selector.setEnabled(True)

        self.update_preview()

    def init_ui_for_data(self):
        numeric_cols = self.df.select_dtypes(include="number").columns
        self.column_selector.clear()
        self.column_selector.addItems(numeric_cols)
        self.update_preview()