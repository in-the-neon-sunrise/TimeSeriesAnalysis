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

    # Загрузка CSV

    def load_data(self, file_path: str):
        if not file_path:
            return

        try:
            df = load_csv(file_path)

            # сохраняем в ProjectService
            self.project.set_raw_data(df, file_path)
            self.data_loaded.emit(df)
            # строим профиль
            profile = build_data_profile(df)
            # уведомляем UI
            self.info_changed.emit(
                f"Файл загружен:\n{file_path}\nСтрок: {len(df)}"
            )

            self.data_loaded.emit(df)
            self.profile_ready.emit(profile)

        except Exception as e:
            self.error_occurred.emit(str(e))


    def get_raw_data(self):
        return self.project.raw_data

    def has_raw_data(self):
        return self.project.has_raw_data()