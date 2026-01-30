from PySide6.QtCore import QObject, Signal
from core.models.project import Project
from infrastructure.project_repository import ProjectRepository
from services.data_service import DataService

class BaseViewModel(QObject):
    error_occurred = Signal(str)
    info_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.project = Project()
        self.data_service = DataService()
        self.repository = ProjectRepository()

    def load_csv(self, path: str):
        df = self.data_service.load_csv(path)
        self.project.csv_path = path
        self.project.raw_data = df