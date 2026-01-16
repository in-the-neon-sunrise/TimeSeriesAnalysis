from PySide6.QtWidgets import QLabel, QVBoxLayout
from ui.pages.base_page import BasePage

class SegmentationPage(BasePage):
    def __init__(self, project):
        super().__init__()
        self.project = project

        layout = QVBoxLayout(self)
        title = QLabel("Сегментация")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")

        layout.addWidget(title)
        layout.addStretch()