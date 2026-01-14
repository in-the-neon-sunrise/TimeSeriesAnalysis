from PySide6.QtCore import Signal

from viewmodels.base_vm import BaseViewModel
from infrastructure.csv_loader import load_csv
from core.data_models.data_profile import build_data_profile

class DataViewModel(BaseViewModel):
    data_loaded = Signal(object)
    profile_ready = Signal(list)

    def __init__(self, project_service):
        super().__init__()
        self.project = project_service

    def load_data(self, file_path: str):
        if not file_path:
            return

        try:
            data = load_csv(file_path)
            self.project.set_raw_data(data, file_path)

            profile = build_data_profile(data)

            self.info_changed.emit(
                f"Файл загружен:\n{file_path}\nСтрок: {len(data)}"
            )
            self.data_loaded.emit(data)
            self.profile_ready.emit(profile)

        except Exception as e:
            self.error_occurred.emit(str(e))