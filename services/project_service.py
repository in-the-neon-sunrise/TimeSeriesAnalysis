from infrastructure.project_repository import ProjectRepository
from core.data_models.project_state import ProjectState


class ProjectService:

    def __init__(self):
        self.repository = ProjectRepository()
        self.state = ProjectState()

    def new_project(self):
        self.state = ProjectState()

    def open_project(self, path: str):
        self.state = self.repository.load_project(path)

    def save(self):
        if not self.state.file_path:
            raise RuntimeError("File path is not set")
        self.repository.save_project(self.state, self.state.file_path)
        self.state.is_dirty = False

    def save_as(self, path: str):
        self.state.file_path = path
        self.save()