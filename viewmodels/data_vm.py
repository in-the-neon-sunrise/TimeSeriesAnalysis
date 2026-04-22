from PySide6.QtCore import Signal

from viewmodels.base_vm import BaseViewModel
from infrastructure.csv_loader import load_csv
from core.data_models.data_profile import build_data_profile

class DataViewModel(BaseViewModel):
    data_loaded = Signal(object)
    profile_ready = Signal(list)

    def __init__(self, project_service, repository):
        super().__init__()
        self.project = project_service
        self.repository = repository

    # Загрузка CSV

    def load_data(self, file_path: str):
        if not file_path:
            return

        try:
            df = load_csv(file_path)

            # сохраняем в ProjectService
            self.project.set_raw_data(df, file_path)
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

    def save_project(self):
        if not self.project.file_path:
            self.error_occurred.emit("Файл проекта не выбран.")
            return

        try:
            self.repository.save(self.project, self.project.file_path)
            self.info_changed.emit("Проект успешно сохранён.")
        except Exception as e:
            self.error_occurred.emit(str(e))

    def save_project_as(self, file_path: str):
        try:
            self.project.file_path = file_path
            self.repository.save(self.project, file_path)
            self.info_changed.emit("Проект успешно сохранён.")
        except Exception as e:
            self.error_occurred.emit(str(e))

    def load_project(self, file_path: str):
        try:
            self.repository.load(file_path, self.project)
            self.project.file_path = file_path

            # уведомляем UI, что данные появились
            if self.project.raw_data is not None:
                self.data_loaded.emit(self.project.raw_data)

            self.info_changed.emit("Проект успешно открыт.")

        except Exception as e:
            self.error_occurred.emit(str(e))