from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QHeaderView, QScrollArea, QGroupBox, QComboBox,
    QHBoxLayout, QMessageBox, QTableView
)
from PySide6.QtCore import Qt

from ui.pages.base_page import BasePage
from ui.models.pandas_table_model import PandasTableModel


class DataPage(BasePage):
    def __init__(self, data_vm):
        super().__init__()

        self.vm = data_vm
        self.vm.info_changed.connect(self.update_info)
        self.vm.error_occurred.connect(self.show_error)

        self.df = None
        self.time_column = None

        # ===============================
        # Scroll container
        # ===============================

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setSpacing(16)

        # ===============================
        # Заголовок
        # ===============================

        title = QLabel("Входные данные")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        # ===============================
        # Информация + загрузка
        # ===============================

        self.info_label = QLabel("Файл не загружен")

        self.load_btn = QPushButton("Загрузить CSV")
        self.load_btn.clicked.connect(self.on_load_clicked)

        layout.addWidget(self.info_label)
        layout.addWidget(self.load_btn)

        # Блок структуры данных

        self.structure_box = QGroupBox("Структура данных")
        self.structure_box.hide()

        struct_layout = QVBoxLayout(self.structure_box)

        self.shape_label = QLabel()
        self.numeric_label = QLabel()
        self.numeric_label.setWordWrap(True)

        struct_layout.addWidget(self.shape_label)
        struct_layout.addWidget(self.numeric_label)

        layout.addWidget(self.structure_box)

        # ===============================
        # Блок временной оси
        # ===============================

        self.time_box = QGroupBox("Временная ось")
        self.time_box.hide()

        time_layout = QVBoxLayout(self.time_box)

        row_layout = QHBoxLayout()

        self.time_selector = QComboBox()
        self.time_selector.currentTextChanged.connect(
            self.on_time_column_changed
        )

        self.no_time_btn = QPushButton("Продолжить без временной оси")
        self.no_time_btn.clicked.connect(self.clear_time_column)

        row_layout.addWidget(self.time_selector)
        row_layout.addWidget(self.no_time_btn)

        self.time_status_label = QLabel()
        self.sort_btn = QPushButton("Отсортировать по времени")
        self.sort_btn.clicked.connect(self.sort_by_time)
        self.sort_btn.hide()

        time_layout.addLayout(row_layout)
        time_layout.addWidget(self.time_status_label)
        time_layout.addWidget(self.sort_btn)

        layout.addWidget(self.time_box)

        # ===============================
        # Предпросмотр
        # ===============================

        self.data_title = QLabel("Предпросмотр данных")
        self.data_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.data_title.hide()

        self.table = QTableView()
        self.table.hide()

        layout.addWidget(self.data_title)
        layout.addWidget(self.table)

        layout.addStretch()

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(scroll)

        self.vm.data_loaded.connect(self.show_data)

    # Загрузка файла

    def on_load_clicked(self):
        if self.vm.has_raw_data():
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

        if not file_path:
            return

        self.vm.load_data(file_path)

    # Отображение данных

    def show_data(self, df):
        self.df = df

        # Таблица
        self.table.setModel(PandasTableModel(df))
        self.table.show()
        self.data_title.show()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.adjust_table_columns()

        # Структура
        self.structure_box.show()
        self.shape_label.setText(
            f"Размер данных: {df.shape[0]} строк × {df.shape[1]} столбцов"
        )

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if numeric_cols:
            self.numeric_label.setText(
                f"Числовые столбцы ({len(numeric_cols)}): "
                + ", ".join(numeric_cols)
            )
        else:
            self.numeric_label.setText("Числовые столбцы не обнаружены")

        # Временная ось
        self.setup_time_detection()

    # Определение временной оси

    def setup_time_detection(self):
        self.time_box.show()
        self.time_selector.clear()

        columns = self.df.columns.tolist()
        self.time_selector.addItems(columns)

        detected = self.detect_time_column()

        if detected:
            index = columns.index(detected)
            self.time_selector.setCurrentIndex(index)
            self.time_column = detected
        else:
            self.time_column = None

        self.update_time_status()

    def detect_time_column(self):
        candidates = []

        for col in self.df.columns:
            series = self.df[col]

            # 1. Уже datetime
            if str(series.dtype).startswith("datetime"):
                candidates.append(col)
                continue

            # 2. Название
            if any(word in col.lower() for word in ["date", "time"]):
                candidates.append(col)

        return candidates[0] if candidates else None

    def on_time_column_changed(self, column_name):
        self.time_column = column_name
        self.update_time_status()

    def clear_time_column(self):
        self.time_column = None
        self.time_status_label.setText(
            "Временная ось не используется."
        )
        self.sort_btn.hide()

    # Проверка упорядоченности

    def update_time_status(self):
        if not self.time_column:
            return

        series = self.df[self.time_column]

        try:
            parsed = series
            is_sorted = parsed.is_monotonic_increasing
        except Exception:
            self.time_status_label.setText(
                "Невозможно проверить упорядоченность."
            )
            self.sort_btn.hide()
            return

        if is_sorted:
            self.time_status_label.setText(
                "✔ Временные метки упорядочены."
            )
            self.sort_btn.hide()
        else:
            self.time_status_label.setText(
                "⚠ Временные метки НЕ упорядочены."
            )
            self.sort_btn.show()

    def sort_by_time(self):
        if not self.time_column:
            return

        self.df = self.df.sort_values(by=self.time_column)
        self.table.setModel(PandasTableModel(self.df))

        self.time_status_label.setText(
            "✔ Данные отсортированы по времени."
        )
        self.sort_btn.hide()

    # Вспомогательные

    def update_info(self, text: str):
        self.info_label.setText(text)

    def show_error(self, message: str):
        QMessageBox.critical(self, "Ошибка", message)

    def adjust_table_columns(self):
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)

        total_width = 0
        for i in range(header.count()):
            total_width += header.sectionSize(i)+2

        viewport_width = self.table.viewport().width()
        print(viewport_width)
        print(total_width)

        # Если суммарная ширина меньше ширины таблицы → растягиваем
        if total_width < viewport_width:
            header.setSectionResizeMode(QHeaderView.Stretch)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "table"):
            self.adjust_table_columns()