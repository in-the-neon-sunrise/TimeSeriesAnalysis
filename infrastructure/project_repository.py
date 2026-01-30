class ProjectRepository:
    def save(self, project, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError