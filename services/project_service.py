import pandas as pd
from typing import Optional, Dict, Any


class ProjectService:
    def __init__(self):
        self.raw_data: Optional[pd.DataFrame] = None

        # Результаты этапов
        self.processed_data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.segments: Optional[pd.DataFrame] = None
        self.clusters: Optional[pd.DataFrame] = None
        self.clustering_result: Optional[Dict[str, Any]] = None
        self.markov_matrix: Optional[pd.DataFrame] = None
        self.markov_result: Optional[dict] = None

        # Параметры шагов
        self.parameters: Dict[str, dict] = {}

        # Служебное
        self.file_path: Optional[str] = None
        self.last_report_path: Optional[str] = None
        self.last_report_generated_at: Optional[str] = None

    def set_raw_data(self, df: pd.DataFrame, file_path=None):
        self.raw_data = df
        self.file_path = file_path

        # При загрузке новых данных сбрасываем результаты
        self.processed_data = None
        self.features = None
        self.segments = None
        self.clusters = None
        self.clustering_result = None
        self.markov_matrix = None
        self.markov_result = None
        self.parameters.clear()
        self.last_report_path = None
        self.last_report_generated_at = None

    def set_processed_data(self, df: pd.DataFrame, params: dict = None):
        self.processed_data = df
        if params:
            self.parameters["preprocessing"] = params

    def set_features(self, df: pd.DataFrame, params: dict = None):
        self.features = df
        if params:
            self.parameters["features"] = params

    def set_segments(self, df: Optional[pd.DataFrame], params: dict = None):
        self.segments = df
        if params:
            self.parameters["segmentation"] = params

    def set_clusters(self, df: Optional[pd.DataFrame], params: dict = None):
        self.clusters = df
        if params:
            self.parameters["clustering"] = params

    def set_markov_matrix(self, df: pd.DataFrame, params: dict = None):
        self.markov_matrix = df
        if params:
            self.parameters["markov"] = params

    def set_markov_result(self, result: dict, params: dict = None):
        self.markov_result = result
        if params:
            self.parameters["markov"] = params

    # проверки

    def has_raw_data(self):
        return self.raw_data is not None

    def has_processed_data(self):
        return self.processed_data is not None
