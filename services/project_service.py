from core.models.project import Project


class ProjectService:
    def __init__(self):
        self.raw_data = None
        self.file_path = None
        self.project = Project()

    def set_raw_data(self, data, file_path):
        self.raw_data = data
        self.file_path = file_path

    def has_data(self):
        return self.project.has_data()

    def replace_project(self, new_project: Project):
        print("Replacing Project 1")
        self.project.csv_path = new_project.csv_path
        self.project.dataframe = new_project.dataframe