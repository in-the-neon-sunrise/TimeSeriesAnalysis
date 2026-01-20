from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QStackedWidget, QFileDialog, QMessageBox
)
from services.project_io import save_project, load_project
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
        self.pages = {}

        self.setWindowTitle("Анализ сейсмологических данных")
        self.resize(1000, 600)

        self.project = project

        self.stack = QStackedWidget()

        data_page = DataPage(self.project)
        preprocessing_page = PreprocessingPage(self.project)
        features_page = FeaturesPage(self.project)
        segmentation_page = SegmentationPage(self.project)
        clustering_page = ClusteringPage(self.project)
        markov_page = MarkovPage(self.project)
        report_page = ReportPage(self.project)

        self.pages["data"] = data_page
        self.pages["preprocessing"] = preprocessing_page
        self.pages["features"] = features_page
        self.pages["segmentation"] = segmentation_page
        self.pages["clustering"] = clustering_page
        self.pages["markov"] = markov_page
        self.pages["report"] = report_page

        self.stack.addWidget(data_page)
        self.stack.addWidget(preprocessing_page)
        self.stack.addWidget(features_page)
        self.stack.addWidget(segmentation_page)
        self.stack.addWidget(clustering_page)
        self.stack.addWidget(markov_page)
        self.stack.addWidget(report_page)

        self.navigation = NavigationPanel(self)

        #self.navigation.data_clicked.connect(self.show_data_page)
        self.navigation.data_clicked.connect(
            lambda: self.show_page(data_page)
        )
        self.navigation.preprocessing_clicked.connect(
            lambda: self.show_page(preprocessing_page)
        )
        self.navigation.features_clicked.connect(
            lambda: self.show_page(features_page)
        )
        self.navigation.segmentation_clicked.connect(
            lambda: self.show_page(segmentation_page)
        )
        self.navigation.clustering_clicked.connect(
            lambda: self.show_page(clustering_page)
        )
        self.navigation.markov_clicked.connect(
            lambda: self.show_page(markov_page)
        )
        self.navigation.report_clicked.connect(
            lambda: self.show_page(report_page)
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
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Файл")

        save_action = file_menu.addAction("Сохранить")
        save_as_action = file_menu.addAction("Сохранить как...")
        open_action = file_menu.addAction("Открыть проект")

        save_action.triggered.connect(self.on_save)
        save_as_action.triggered.connect(self.on_save_as)
        open_action.triggered.connect(self.on_open)

    def on_save_as(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить проект",
            "",
            "Project Files (*.json)"
        )
        if not file_path:
            return
        if not file_path.endswith(".json"):
            file_path += ".json"

        save_project(self.project.project, file_path)
        self.current_project_path = file_path

    def on_save(self):
        if hasattr(self, "current_project_path"):
            save_project(self.project.project, self.current_project_path)
        else:
            self.on_save_as()

    def on_open(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть проект",
            "",
            "Project Files (*.json)"
        )
        if not file_path:
            return

        reply = QMessageBox.question(
            self,
            "Подтверждение",
            "Текущий проект будет закрыт. Продолжить?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        loaded_project = load_project(file_path)
        self.project.replace_project(loaded_project)
        self.current_project_path = file_path

        self.refresh_ui_after_project_load()

    def refresh_ui_after_project_load(self):
        for page in self.pages.values():
            if hasattr(page, "on_enter"):
                page.on_enter()

