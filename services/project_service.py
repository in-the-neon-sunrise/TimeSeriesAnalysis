class ProjectService:
    def __init__(self):
        self.raw_data = None
        self.file_path = None

    def set_raw_data(self, data, file_path):
        self.raw_data = data
        self.file_path = file_path

    def has_data(self):
        return self.raw_data is not None