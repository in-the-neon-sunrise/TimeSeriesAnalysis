from PySide6.QtWidgets import QLabel, QVBoxLayout
from ui.pages.base_page import BasePage

class MarkovPage(BasePage):
    def __init__(self, data_vm):
        super().__init__()
        self.vm = data_vm

        layout = QVBoxLayout(self)
        title = QLabel("Цепи Маркова")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")

        layout.addWidget(title)
        layout.addStretch()