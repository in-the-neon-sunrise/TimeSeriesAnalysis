from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QHeaderView, QScrollArea, QGroupBox, QComboBox
)
from PySide6.QtWidgets import QRadioButton, QHBoxLayout

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from ui.pages.base_page import BasePage
from viewmodels.data_vm import DataViewModel
from PySide6.QtWidgets import QTableView, QMessageBox
from ui.models.pandas_table_model import PandasTableModel
from ui.models.profile_table_model import ProfileTableModel
from services.data_statistics_service import DataStatisticsService
from matplotlib.figure import Figure

class DataPage(BasePage):
    def __init__(self, project):
        super().__init__()

        self.vm = project.data_vm
        self.vm.info_changed.connect(self.update_info)
        self.vm.error_occurred.connect(self.show_error)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setSpacing(14)

        #Заголовок
        title = QLabel("Входные данные")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        #Информация
        self.info_label = QLabel("Файл не загружен")
        self.load_btn = QPushButton("Загрузить CSV")
        self.load_btn.clicked.connect(self.on_load_clicked)

        layout.addWidget(self.info_label)
        layout.addWidget(self.load_btn)

        #Предпросмотр
        self.data_title = QLabel("Предпросмотр данных")
        self.data_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.data_title.hide()

        self.table = QTableView()
        self.table.hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout.addWidget(self.data_title)
        layout.addWidget(self.table)

        #Характеристики
        self.profile_title = QLabel("Характеристики столбцов")
        self.profile_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.profile_title.hide()

        self.profile_table = QTableView()
        self.profile_table.hide()
        self.profile_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout.addWidget(self.profile_title)
        layout.addWidget(self.profile_table)

        #Общая статистика
        self.stats_box = QGroupBox("Общая статистика")
        self.stats_box.hide()
        stats_layout = QVBoxLayout(self.stats_box)

        self.rows_label = QLabel()
        self.missing_label = QLabel()

        stats_layout.addWidget(self.rows_label)
        stats_layout.addWidget(self.missing_label)

        layout.addWidget(self.stats_box)
        layout.addStretch()

        #Диагностика временного ряда
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

        #ADF
        self.adf_label = QLabel()
        self.adf_label.setWordWrap(True)
        diag_layout.addWidget(self.adf_label)

        #ACF график
        self.acf_figure = Figure(figsize=(5, 3))
        self.acf_canvas = FigureCanvas(self.acf_figure)

        acf_container = QWidget()
        acf_layout = QVBoxLayout(acf_container)
        acf_layout.setContentsMargins(0, 0, 0, 0)
        acf_layout.addWidget(self.acf_canvas)

        diag_layout.addWidget(acf_container)

        layout.addWidget(self.diagnostics_box)

        #Выбросы
        outliers_title = QLabel("Выбросы")
        outliers_title.setStyleSheet("font-weight: bold; margin-top: 10px;")
        diag_layout.addWidget(outliers_title)

        # выбор метода
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

        # количество выбросов
        self.outliers_label = QLabel()
        diag_layout.addWidget(self.outliers_label)

        # boxplot
        self.boxplot_figure = Figure(figsize=(5, 2.5))
        self.boxplot_canvas = FigureCanvas(self.boxplot_figure)

        boxplot_container = QWidget()
        boxplot_layout = QVBoxLayout(boxplot_container)
        boxplot_layout.setContentsMargins(0, 0, 0, 0)
        boxplot_layout.addWidget(self.boxplot_canvas)

        diag_layout.addWidget(boxplot_container)

        #Всякое
        root_layout = QVBoxLayout(self)
        root_layout.addWidget(scroll)

        self.vm.data_loaded.connect(self.show_data)
        self.vm.profile_ready.connect(self.show_profile)

    def on_enter(self):
        pass

    def on_load_clicked(self):
        if self.vm.project.has_data():
            reply = QMessageBox.question(
                self,
                "Подтверждение",
                "Загрузить новый файл?\nТекущие данные будут заменены.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выбор CSV", "", "CSV Files (*.csv)"
        )
        self.vm.load_data(file_path)

        self.table.show()
        self.profile_table.show()
        self.data_title.show()
        self.profile_title.show()

    def update_info(self, text: str):
        self.info_label.setText(text)

    def show_error(self, message: str):
        QMessageBox.critical(self, "Ошибка", message)

    def show_data(self, df):
        self.df = df  # сохраним ссылку
        self.table.setModel(PandasTableModel(df))
        self.table.show()
        self.data_title.show()

        info = DataStatisticsService.basic_info(df)

        self.rows_label.setText(
            f"Количество наблюдений: {info['rows']}"
        )
        self.missing_label.setText(
            f"Доля пропусков: {info['missing_ratio']:.2%}"
        )

        self.stats_box.show()

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        self.column_selector.clear()
        self.column_selector.addItems(numeric_cols)

        if numeric_cols:
            self.diagnostics_box.show()
            self.update_diagnostics(numeric_cols[0])

    def show_profile(self, profile):
        self.profile_table.setModel(ProfileTableModel(profile))
        self.profile_table.show()
        self.profile_title.show()
        self.load_btn.setText("Заменить файл")

    def on_column_changed(self, column_name: str):
        if not column_name:
            return
        self.update_diagnostics(column_name)

    def update_diagnostics(self, column_name: str):
        series = self.df[column_name]

        #ADF
        adf = DataStatisticsService.stationarity_adf(series)

        adf_text = (
            f"<b>ADF-тест</b><br>"
            f"ADF statistic: {adf['adf_statistic']:.4f}<br>"
            f"p-value: {adf['p_value']:.4f}<br>"
            f"Стационарность: "
            f"<b>{'ДА' if adf['is_stationary'] else 'НЕТ'}</b>"
        )

        self.adf_label.setText(adf_text)

        #ACF
        acf_values = DataStatisticsService.autocorrelation(series)

        self.acf_figure.clear()
        ax = self.acf_figure.add_subplot(111)

        ax.stem(acf_values)
        ax.set_title("Autocorrelation Function (ACF)")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")

        self.acf_canvas.draw()

        self.update_outliers(column_name)

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