from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QStackedWidget
)
from ui.navigation_panel import NavigationPanel
from ui.pages.data_page import DataPage

class MainWindow(QMainWindow):
    def __init__(self, project):
        super().__init__()
        self.setWindowTitle("Анализ сейсмологических данных")
        self.resize(1000, 600)

        self.project = project

        self.stack = QStackedWidget()

        self.data_page = DataPage(self.project)
        self.stack.addWidget(self.data_page)

        self.navigation = NavigationPanel(self)
        self.navigation.data_clicked.connect(self.show_data_page)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.navigation)
        layout.addWidget(self.stack)

        self.setCentralWidget(central)

    def show_data_page(self):
        self.stack.setCurrentWidget(self.data_page)