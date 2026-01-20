from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QHeaderView
)
from ui.pages.base_page import BasePage
from viewmodels.data_vm import DataViewModel
from PySide6.QtWidgets import QTableView, QMessageBox
from ui.models.pandas_table_model import PandasTableModel
from ui.models.profile_table_model import ProfileTableModel

class DataPage(BasePage):
    def __init__(self, project):
        super().__init__()

        self.vm = DataViewModel(project)

        self.vm.info_changed.connect(self.update_info)
        self.vm.error_occurred.connect(self.show_error)

        self.info_label = QLabel("Файл не загружен")
        self.load_btn = QPushButton("Загрузить CSV")
        self.load_btn.clicked.connect(self.on_load_clicked)

        self.table = QTableView()
        self.profile_table = QTableView()
        self.table.hide()
        self.profile_table.hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.profile_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.vm.data_loaded.connect(self.show_data)
        self.vm.profile_ready.connect(self.show_profile)

        self.data_title = QLabel("Предпросмотр данных")
        self.data_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.data_title.hide()

        self.profile_title = QLabel("Характеристики столбцов")
        self.profile_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.profile_title.hide()

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Входные данные")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")

        layout.addWidget(title)
        layout.addWidget(self.info_label)
        layout.addWidget(self.load_btn)

        layout.addSpacing(10)

        layout.addWidget(self.data_title)
        layout.addWidget(self.table)

        layout.addSpacing(10)

        layout.addWidget(self.profile_title)
        layout.addWidget(self.profile_table)

        layout.addStretch()

    def on_enter(self):
        if self.vm.project.has_data():
            print("Replacing Project 2")
            self.show_data(self.vm.project.dataframe)
            self.vm.load_data(self.vm.project.csv_path)
            self.load_btn.setText("Заменить файл")
            self.info_label.setText("Файл загружен из проекта")

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
        self.table.setModel(PandasTableModel(df))

    def show_profile(self, profile):
        self.profile_table.setModel(ProfileTableModel(profile))
        self.load_btn.setText("Заменить файл")