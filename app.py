from ui.main_window import MainWindow
from services.project_service import ProjectService

class Application:
    def __init__(self):
        self.project = ProjectService()
        self.main_window = MainWindow(self.project)

    def run(self):
        self.main_window.show()