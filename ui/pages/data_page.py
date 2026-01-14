from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox
)
from viewmodels.data_vm import DataViewModel
from PySide6.QtWidgets import QTableView, QMessageBox
from ui.models.pandas_table_model import PandasTableModel
from ui.models.profile_table_model import ProfileTableModel

class DataPage(QWidget):
    def __init__(self, project):
        super().__init__()

        self.vm = DataViewModel(project)

        self.vm.info_changed.connect(self.update_info)
        self.vm.error_occurred.connect(self.show_error)

        layout = QVBoxLayout(self)

        self.info_label = QLabel("Файл не загружен")
        self.load_btn = QPushButton("Загрузить CSV")
        self.load_btn.clicked.connect(self.on_load_clicked)

        layout.addWidget(self.info_label)
        layout.addWidget(self.load_btn)
        layout.addStretch()

        self.table = QTableView()
        self.profile_table = QTableView()

        layout.addWidget(self.table)
        layout.addWidget(self.profile_table)

        self.vm.data_loaded.connect(self.show_data)
        self.vm.profile_ready.connect(self.show_profile)

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

    def update_info(self, text: str):
        self.info_label.setText(text)

    def show_error(self, message: str):
        QMessageBox.critical(self, "Ошибка", message)

    def show_data(self, df):
        self.table.setModel(PandasTableModel(df))

    def show_profile(self, profile):
        self.profile_table.setModel(ProfileTableModel(profile))
        self.load_btn.setText("Заменить файл")