import os
import random
import typing

import numpy
import pandas
import gtda.plotting
import gtda.homology
import gtda.time_series

from .FeatureCalculator import FeatureCalculator

def set_random_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class OverallFeatureExtractor:
    def __init__(
        self,

        dimension: int = 5,
        time_delay: int = 11,
        stride: int = 3,
        
        homology_dimensions = [ 1, 2, 3 ],
        
        reduced: bool = False,
        filtering_percentile: int = 10,

        n_jobs: int = -1,
        random_state: int = 42,
        print_obj: typing.Optional[int] = 0,
        folder: typing.Optional[str] = None
    ):
        self.dimension = dimension
        self.time_delay = time_delay
        self.stride = stride

        self.homology_dimensions = homology_dimensions

        self.reduced = reduced
        self.filtering_percentile = filtering_percentile

        self.n_jobs = n_jobs
        self.print_obj = print_obj
        self.random_state = random_state
        
        self.diagrams_file = f'{folder}/diagrams.npy' if folder else None
        self.features_file = f"{folder}/features.feather" if folder else None
        
        self.point_cloud_image = f"{folder}/point_cloud.svg" if folder and self.print_obj is not None else None
        self.diagram_image = f'{folder}/diagram.svg' if folder and self.print_obj is not None else None


    def get_point_clouds(self, data: numpy.ndarray) -> typing.List[numpy.ndarray]:
        embedding = gtda.time_series.TakensEmbedding(
            dimension = self.dimension,
            time_delay = self.time_delay,
            stride = self.stride,
            flatten = True
        )
        point_clouds = embedding.fit_transform(data)

        if self.point_cloud_image is not None:
            plot = gtda.plotting.plot_point_cloud(point_clouds[self.print_obj])
            plot.write_image(file = self.point_cloud_image, format = "svg")

        print(f'Point clouds: {point_clouds.shape}')
        return point_clouds
    

    def calculate_persistence(self, point_clouds: typing.List[numpy.ndarray]) -> numpy.ndarray:
        if self.diagrams_file is not None and os.path.exists(self.diagrams_file):
            print(f'Got diagrams from {self.diagrams_file}')
            return numpy.load(self.diagrams_file)

        print('Calculating persistence')
        persistence = gtda.homology.VietorisRipsPersistence(homology_dimensions = self.homology_dimensions, n_jobs = self.n_jobs)
        diagrams = persistence.fit_transform(point_clouds)
        print(f'Diagrams: {diagrams.shape}')

        if self.diagram_image is not None and self.print_obj is not None:
            diagram_plot = gtda.plotting.plot_diagram(diagrams[self.print_obj])
            diagram_plot.write_image(file = self.diagram_image, format = "svg")

        if self.diagrams_file is not None:
            print(f'Saving diagrams to {self.diagrams_file}')
            numpy.save(self.diagrams_file, diagrams)
        return diagrams
    

    def calculate_features(self, diagrams: numpy.ndarray) -> pandas.DataFrame:
        if self.features_file is not None and os.path.exists(self.features_file):
            print(f'Got features from {self.features_file}')
            return pandas.read_feather(self.features_file)

        calculator = FeatureCalculator(filtering_percentile = self.filtering_percentile, n_jobs = self.n_jobs, reduced = self.reduced)
        features = calculator.calc_features(diagrams, prefix = 'overall')
        print(f'Features: {features.shape}')
        
        if self.features_file is not None:
            print(f'Saving features to {self.features_file}')
            features.to_feather(self.features_file)
        return features


    def extract(self, data: numpy.ndarray) -> pandas.DataFrame:
        set_random_seed(self.random_state)

        if self.features_file is not None and os.path.exists(self.features_file):
            return self.calculate_features(None)
        
        if self.diagrams_file is not None and os.path.exists(self.diagrams_file):
            diagrams = self.calculate_persistence(None)
            return self.calculate_features(data.shape[1], diagrams)
        
        point_clouds = self.get_point_clouds(data)
        diagrams = self.calculate_persistence(point_clouds)
        return self.calculate_features(diagrams)
