import os
import random
import typing
import collections

import tqdm
import numpy
import pandas
import joblib
import itertools
import gtda.plotting
import gtda.homology
import gtda.time_series

from .FeatureCalculator import FeatureCalculator

def set_random_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class PerChannelFeatureExtractor:
    def __init__(
        self,

        max_time_delay: int = 20,
        max_dimension: int = 20,
        stride: int = 3,
        
        homology_dimensions = [ 1, 2 ],
        
        reduced: bool = False,
        filtering_percentile: int = 10,

        n_jobs: int = -1,
        random_state: int = 42,
        print_obj: typing.Optional[int] = 0,
        folder: typing.Optional[str] = None
    ):
        self.max_time_delay = max_time_delay
        self.max_dimension = max_dimension
        self.stride = stride

        self.homology_dimensions = homology_dimensions

        self.reduced = reduced
        self.filtering_percentile = filtering_percentile

        self.n_jobs = n_jobs
        self.print_obj = print_obj
        self.random_state = random_state

        self.embedding_params_file = f'{folder}/embedders_params.npy' if folder else None
        self.diagrams_file = f'{folder}/diagrams.npy' if folder else None
        self.features_file = f"{folder}/features.feather" if folder else None
        
        self.point_cloud_image = f"{folder}/point_cloud.svg" if folder and self.print_obj is not None else None
        self.diagram_image = f'{folder}/diagram.svg' if folder and self.print_obj is not None else None

    def determine_embedding_params(self, data: numpy.ndarray) -> typing.List[typing.List[typing.Tuple[int, int]]]:
        if self.embedding_params_file is not None and os.path.exists(self.embedding_params_file):
            print(f'Got embedding params from {self.embedding_params_file}')
            return [ list(map(tuple, params_data)) for params_data in numpy.load(self.embedding_params_file) ]
        
        def fit_searcher(i: int, j: int, ts: numpy.ndarray) -> typing.Tuple[int, int, int, int]:
            searcher = gtda.time_series.SingleTakensEmbedding(
                parameters_type = 'search',
                time_delay = self.max_time_delay,
                dimension = self.max_dimension,
                stride = self.stride
            ).fit(ts)
            return (i, j, searcher.dimension_, searcher.time_delay_)

        loop = list(itertools.product(range(data.shape[0]), range(data.shape[1])))
        searchers = joblib.Parallel(return_as = 'generator', n_jobs = self.n_jobs)(
            joblib.delayed(fit_searcher)(i, j, data[i, j, :]) for i, j in loop
        )
        
        embedders_params = [ [ None ] * data.shape[0] for _ in range(data.shape[1]) ]
        for i, j, dimension, time_delay in tqdm.tqdm(searchers, total = len(loop), desc = 'embedding params'):
            embedders_params[j][i] = (dimension, time_delay)
        
        if self.embedding_params_file is not None:
            print(f'Saving embedding params to {self.embedding_params_file}')
            numpy.save(self.embedding_params_file, embedders_params)
        return embedders_params

    
    def get_embedders(
        self,
        embedders_params: typing.List[typing.List[typing.Tuple[int, int]]]
    ) -> typing.List[gtda.time_series.SingleTakensEmbedding]:
        embedders = [ ]
        for i in tqdm.trange(len(embedders_params), desc = 'embedders'):
            (dimension, time_delay), _ = collections.Counter(embedders_params[i]).most_common(1)[0]
            attractor = gtda.time_series.SingleTakensEmbedding(
                parameters_type = 'fixed',
                time_delay = int(time_delay),
                dimension = int(dimension),
                stride = self.stride
            )
            embedders.append(attractor)
        return embedders
    

    def get_point_clouds(
        self,
        data: numpy.ndarray,
        embedders: typing.List[gtda.time_series.SingleTakensEmbedding]
    ) -> typing.List[numpy.ndarray]:
        def get_point_cloud(embedder: gtda.time_series.SingleTakensEmbedding, ts: numpy.ndarray) -> numpy.ndarray:
            return embedder.fit_transform(ts)
        
        loop = list(itertools.product(range(data.shape[0]), range(data.shape[1])))
        point_clouds = joblib.Parallel(return_as = 'generator', n_jobs = self.n_jobs)(
            joblib.delayed(get_point_cloud)(embedders[j], data[i, j, :]) for i, j in loop
        )
        point_clouds = list(tqdm.tqdm(point_clouds, total = len(loop), desc = 'point clouds'))
    
        if self.point_cloud_image is not None and self.print_obj is not None:
            point_cloud_plot = gtda.plotting.plot_point_cloud(point_clouds[self.print_obj])
            point_cloud_plot.write_image(file = self.point_cloud_image, format = "svg")
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
    

    def calculate_features(self, n_channels: int, diagrams: numpy.ndarray) -> pandas.DataFrame:
        if self.features_file is not None and os.path.exists(self.features_file):
            print(f'Got features from {self.features_file}')
            return pandas.read_feather(self.features_file)
        calculator = FeatureCalculator(filtering_percentile = self.filtering_percentile, n_jobs = self.n_jobs, reduced = self.reduced)
        features_raw = calculator.calc_features(diagrams)
        features = pandas.concat([ features_raw.iloc[i::n_channels, :].reset_index(drop = True) for i in range(n_channels) ], axis = 1)
        features.columns = [ f"channel-{i}{feature_name}" for i in range(n_channels) for feature_name in features_raw ]
        print(f'Features: {features.shape}')
        
        if self.features_file is not None:
            print(f'Saving features to {self.features_file}')
            features.to_feather(self.features_file)
        return features


    def extract(self, data: numpy.ndarray) -> pandas.DataFrame:
        set_random_seed(self.random_state)

        if self.features_file is not None and os.path.exists(self.features_file):
            return self.calculate_features(None, None)
        
        if self.diagrams_file is not None and os.path.exists(self.diagrams_file):
            diagrams = self.calculate_persistence(None)
            return self.calculate_features(data.shape[1], diagrams)
        
        embedders_params = self.determine_embedding_params(data)
        embedders = self.get_embedders(embedders_params)
        point_clouds = self.get_point_clouds(data, embedders)
        diagrams = self.calculate_persistence(point_clouds)
        return self.calculate_features(data.shape[1], diagrams)
