from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Signal

class NavigationPanel(QWidget):
    data_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        self.data_btn = QPushButton("Импорт данных")
        self.data_btn.clicked.connect(self.data_clicked.emit)

        layout.addWidget(self.data_btn)
        layout.addStretch()