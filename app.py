from infrastructure.project_repository import ProjectRepository
from ui.main_window import MainWindow
from services.project_service import ProjectService
from viewmodels.data_vm import DataViewModel
from viewmodels.report_vm import ReportViewModel


class Application:
    def __init__(self):
        self.project = ProjectService()
        repository = ProjectRepository()
        data_vm = DataViewModel(self.project, repository)
        report_vm = ReportViewModel(self.project)
        self.main_window = MainWindow(data_vm, report_vm)

    def run(self):
        self.main_window.show()