from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableView,
    QGroupBox, QComboBox, QRadioButton,
    QHBoxLayout, QScrollArea
)
from PySide6.QtWidgets import QHeaderView

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ui.canvas import ScrollFriendlyCanvas
from ui.pages.base_page import BasePage
from ui.models.profile_table_model import ProfileTableModel
from services.data_statistics_service import DataStatisticsService


class PrimaryAnalysisPage(BasePage):
    def __init__(self, project):
        super().__init__()

        self.vm = project.data_vm
        self.df = None

        self.vm.data_loaded.connect(self.on_data_ready)
        self.vm.profile_ready.connect(self.show_profile)

        # --- Scroll ---
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setSpacing(14)

        # Заголовок

        title = QLabel("Первичный анализ данных")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # Общая статистика

        self.stats_box = QGroupBox("Общая статистика")
        self.stats_box.hide()

        stats_layout = QVBoxLayout(self.stats_box)

        self.rows_label = QLabel()
        self.missing_label = QLabel()

        stats_layout.addWidget(self.rows_label)
        stats_layout.addWidget(self.missing_label)

        layout.addWidget(self.stats_box)

        # Характеристики столбцов

        self.profile_title = QLabel("Характеристики столбцов")
        self.profile_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.profile_title.hide()

        self.profile_table = QTableView()
        self.profile_table.hide()
        self.profile_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        layout.addWidget(self.profile_title)
        layout.addWidget(self.profile_table)

        # Диагностика временного ряда

        self.diagnostics_box = QGroupBox("Диагностика временного ряда")
        self.diagnostics_box.hide()

        diag_layout = QVBoxLayout(self.diagnostics_box)

        # выбор столбца
        self.column_selector = QComboBox()
        self.column_selector.currentTextChanged.connect(
            self.on_column_changed
        )

        diag_layout.addWidget(QLabel("Анализируемый столбец:"))
        diag_layout.addWidget(self.column_selector)

        # ---------- ADF ----------
        self.adf_label = QLabel()
        self.adf_label.setWordWrap(True)
        diag_layout.addWidget(self.adf_label)

        # ---------- ACF ----------
        self.acf_figure = Figure(figsize=(5, 3))
        self.acf_canvas = ScrollFriendlyCanvas(self.acf_figure)

        acf_container = QWidget()
        acf_layout = QVBoxLayout(acf_container)
        acf_layout.setContentsMargins(0, 0, 0, 0)
        acf_layout.addWidget(self.acf_canvas)

        diag_layout.addWidget(acf_container)

        # Выбросы

        outliers_title = QLabel("Выбросы")
        outliers_title.setStyleSheet("font-weight: bold; margin-top: 10px;")
        diag_layout.addWidget(outliers_title)

        method_layout = QHBoxLayout()
        self.iqr_radio = QRadioButton("IQR")
        self.zscore_radio = QRadioButton("Z-score")
        self.iqr_radio.setChecked(True)

        self.iqr_radio.toggled.connect(self.on_outlier_method_changed)
        self.zscore_radio.toggled.connect(self.on_outlier_method_changed)

        method_layout.addWidget(self.iqr_radio)
        method_layout.addWidget(self.zscore_radio)
        method_layout.addStretch()

        diag_layout.addLayout(method_layout)

        self.outliers_label = QLabel()
        diag_layout.addWidget(self.outliers_label)

        # boxplot
        self.boxplot_figure = Figure(figsize=(5, 2.5))
        self.boxplot_canvas = ScrollFriendlyCanvas(self.boxplot_figure)

        boxplot_container = QWidget()
        boxplot_layout = QVBoxLayout(boxplot_container)
        boxplot_layout.setContentsMargins(0, 0, 0, 0)
        boxplot_layout.addWidget(self.boxplot_canvas)

        diag_layout.addWidget(boxplot_container)

        layout.addWidget(self.diagnostics_box)

        layout.addStretch()

        # root layout
        root_layout = QVBoxLayout(self)
        root_layout.addWidget(scroll)


    def on_enter(self):
        if self.vm.project.has_data():
            self.setup_analysis(self.vm.project.get_data())

    # ==========================================================
    # Data handling
    # ==========================================================

    def on_data_ready(self, df):
        self.setup_analysis(df)

    def setup_analysis(self, df):
        self.df = df

        # --- basic info ---
        info = DataStatisticsService.basic_info(df)

        self.rows_label.setText(
            f"Количество наблюдений: {info['rows']}"
        )
        self.missing_label.setText(
            f"Доля пропусков: {info['missing_ratio']:.2%}"
        )

        self.stats_box.show()

        # --- numeric columns ---
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        self.column_selector.clear()
        self.column_selector.addItems(numeric_cols)

        if numeric_cols:
            self.diagnostics_box.show()
            self.update_diagnostics(numeric_cols[0])

    # ==========================================================
    # Profile table
    # ==========================================================

    def show_profile(self, profile):
        self.profile_table.setModel(ProfileTableModel(profile))
        self.profile_table.show()
        self.profile_title.show()
        self.adjust_table_columns()

    # ==========================================================
    # Diagnostics
    # ==========================================================

    def on_column_changed(self, column_name: str):
        if not column_name:
            return
        self.update_diagnostics(column_name)

    def update_diagnostics(self, column_name: str):
        if self.df is None:
            return

        series = self.df[column_name]

        # ---------- ADF ----------
        adf = DataStatisticsService.stationarity_adf(series)

        adf_text = (
            f"<b>ADF-тест</b><br>"
            f"ADF statistic: {adf['adf_statistic']:.4f}<br>"
            f"p-value: {adf['p_value']:.4f}<br>"
            f"Стационарность: "
            f"<b>{'ДА' if adf['is_stationary'] else 'НЕТ'}</b>"
        )

        self.adf_label.setText(adf_text)

        # ---------- ACF ----------
        acf_values = DataStatisticsService.autocorrelation(series)

        self.acf_figure.clear()
        ax = self.acf_figure.add_subplot(111)

        ax.stem(acf_values)
        ax.set_title("Autocorrelation Function (ACF)")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")

        self.acf_canvas.draw()

        # ---------- Outliers ----------
        self.update_outliers(column_name)

    # ==========================================================
    # Outliers
    # ==========================================================

    def on_outlier_method_changed(self):
        column = self.column_selector.currentText()
        if column:
            self.update_outliers(column)

    def update_outliers(self, column_name: str):
        series = self.df[column_name].dropna()

        if self.iqr_radio.isChecked():
            result = DataStatisticsService.detect_outliers_iqr(series)
            method_name = "IQR"
        else:
            result = DataStatisticsService.detect_outliers_zscore(series)
            method_name = "Z-score"

        self.outliers_label.setText(
            f"Метод: {method_name} | "
            f"Количество выбросов: {result['count']}"
        )

        self.boxplot_figure.clear()
        ax = self.boxplot_figure.add_subplot(111)

        ax.boxplot(series, vert=False)
        ax.set_title("Boxplot временного ряда")
        ax.set_xlabel(column_name)

        self.boxplot_canvas.draw()

    def adjust_table_columns(self):
        header = self.profile_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)

        total_width = 0
        for i in range(header.count()):
            total_width += header.sectionSize(i)+2

        viewport_width = self.profile_table.viewport().width()
        print(viewport_width)
        print(total_width)

        # Если суммарная ширина меньше ширины таблицы → растягиваем
        if total_width < viewport_width:
            header.setSectionResizeMode(QHeaderView.Stretch)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "table"):
            self.adjust_table_columns()