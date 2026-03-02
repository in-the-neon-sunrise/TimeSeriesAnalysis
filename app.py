from ui.main_window import MainWindow
from services.project_service import ProjectService
from viewmodels.data_vm import DataViewModel


class Application:
    def __init__(self):
        self.project = ProjectService()
        data_vm = DataViewModel(self.project)
        self.main_window = MainWindow(data_vm)

    def run(self):
        self.main_window.show()