from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
)
from infrastructure.csv_loader import load_csv

class DataPage(QWidget):
    def __init__(self, project):
        super().__init__()
        self.project = project

        layout = QVBoxLayout(self)

        self.info_label = QLabel("Файл не загружен")

        self.load_btn = QPushButton("Загрузить CSV")
        self.load_btn.clicked.connect(self.load_data)

        layout.addWidget(self.info_label)
        layout.addWidget(self.load_btn)
        layout.addStretch()

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выбор CSV", "", "CSV Files (*.csv)"
        )

        if not file_path:
            return

        data = load_csv(file_path)
        self.project.set_raw_data(data, file_path)

        self.info_label.setText(
            f"Загружен файл:\n{file_path}\nСтрок: {len(data)}"
        )