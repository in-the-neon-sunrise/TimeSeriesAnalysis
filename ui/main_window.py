from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QStackedWidget, QFileDialog
)
from ui.navigation_panel import NavigationPanel
from ui.pages.data_page import DataPage
from ui.pages.preprocessing_page import PreprocessingPage
from ui.pages.features_page import FeaturesPage
from ui.pages.primary_analysis_page import PrimaryAnalysisPage
from ui.pages.segmentation_page import SegmentationPage
from ui.pages.clustering_page import ClusteringPage
from ui.pages.markov_page import MarkovPage
from ui.pages.report_page import ReportPage
from PySide6.QtGui import QAction
from PySide6.QtGui import QKeySequence

class MainWindow(QMainWindow):
    def __init__(self, data_vm):
        super().__init__()
        self.setWindowTitle("Анализ сейсмологических данных")
        self.resize(1000, 600)

        self.data_vm = data_vm

        self.stack = QStackedWidget()

        self.data_page = DataPage(self.data_vm)
        self.primary_analysis_page = PrimaryAnalysisPage(self.data_vm)
        self.preprocessing_page = PreprocessingPage(self.data_vm)
        self.features_page = FeaturesPage(self.data_vm)
        self.segmentation_page = SegmentationPage(self.data_vm)
        self.clustering_page = ClusteringPage(self.data_vm)
        self.markov_page = MarkovPage(self.data_vm)
        self.report_page = ReportPage(self.data_vm)

        self.stack.addWidget(self.data_page)
        self.stack.addWidget(self.primary_analysis_page)
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
        self.navigation.primary_analysis_clicked.connect(
            lambda: self.show_page(self.primary_analysis_page)
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

        self._create_menu()

    def show_page(self, page):
        current = self.stack.currentWidget()
        if hasattr(current, "on_leave"):
            current.on_leave()

        self.stack.setCurrentWidget(page)

        if hasattr(page, "on_enter"):
            page.on_enter()

    def _create_menu(self):
        menu_bar = self.menuBar()

        # ===== Меню "Файл" =====
        file_menu = menu_bar.addMenu("Файл")

        # --- Открыть ---
        open_action = QAction("Открыть...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_project)

        # --- Сохранить ---
        save_action = QAction("Сохранить", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_project)

        # --- Сохранить как ---
        save_as_action = QAction("Сохранить как...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self._save_project_as)

        file_menu.addAction(open_action)
        file_menu.addSeparator()
        file_menu.addAction(save_action)
        file_menu.addAction(save_as_action)

    def _open_project(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть проект",
            "",
            "Project Files (*.sqlite *.db)"
        )

        if file_path:
            self.data_vm.load_project(file_path)

    def _save_project(self):
        self.data_vm.save_project()

    def _save_project_as(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить проект как",
            "",
            "Project Files (*.sqlite *.db)"
        )

        if file_path:
            self.data_vm.save_project_as(file_path)
