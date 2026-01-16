from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QStackedWidget
)
from ui.navigation_panel import NavigationPanel
from ui.pages.data_page import DataPage
from ui.pages.preprocessing_page import PreprocessingPage
from ui.pages.features_page import FeaturesPage
from ui.pages.segmentation_page import SegmentationPage
from ui.pages.clustering_page import ClusteringPage
from ui.pages.markov_page import MarkovPage
from ui.pages.report_page import ReportPage

class MainWindow(QMainWindow):
    def __init__(self, project):
        super().__init__()
        self.setWindowTitle("Анализ сейсмологических данных")
        self.resize(1000, 600)

        self.project = project

        self.stack = QStackedWidget()

        self.data_page = DataPage(self.project)
        self.preprocessing_page = PreprocessingPage(self.project)
        self.features_page = FeaturesPage(self.project)
        self.segmentation_page = SegmentationPage(self.project)
        self.clustering_page = ClusteringPage(self.project)
        self.markov_page = MarkovPage(self.project)
        self.report_page = ReportPage(self.project)

        self.stack.addWidget(self.data_page)
        self.stack.addWidget(self.preprocessing_page)
        self.stack.addWidget(self.features_page)
        self.stack.addWidget(self.segmentation_page)
        self.stack.addWidget(self.clustering_page)
        self.stack.addWidget(self.markov_page)
        self.stack.addWidget(self.report_page)

        self.navigation = NavigationPanel(self)

        #self.navigation.data_clicked.connect(self.show_data_page)
        self.navigation.data_clicked.connect(
            lambda: self.show_page(self.data_page)
        )
        self.navigation.preprocessing_clicked.connect(
            lambda: self.show_page(self.preprocessing_page)
        )
        self.navigation.features_clicked.connect(
            lambda: self.show_page(self.features_page)
        )
        self.navigation.segmentation_clicked.connect(
            lambda: self.show_page(self.segmentation_page)
        )
        self.navigation.clustering_clicked.connect(
            lambda: self.show_page(self.clustering_page)
        )
        self.navigation.markov_clicked.connect(
            lambda: self.show_page(self.markov_page)
        )
        self.navigation.report_clicked.connect(
            lambda: self.show_page(self.report_page)
        )

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.navigation)
        layout.addWidget(self.stack)

        self.setCentralWidget(central)

    def show_data_page(self):
        self.stack.setCurrentWidget(self.data_page)

    def show_page(self, page):
        current = self.stack.currentWidget()
        if hasattr(current, "on_leave"):
            current.on_leave()

        self.stack.setCurrentWidget(page)

        if hasattr(page, "on_enter"):
            page.on_enter()

