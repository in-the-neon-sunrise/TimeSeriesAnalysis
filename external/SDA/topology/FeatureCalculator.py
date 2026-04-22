import os
import random

import tqdm
import numpy
import pandas
import joblib
import scipy.stats
import gtda.curves
import gtda.diagrams

def determine_filtering_epsilon(diagrams: numpy.ndarray, percentile: int):
    life = (diagrams[:, :, 1] - diagrams[:, :, 0]).flatten()
    return numpy.percentile(life[life != 0], percentile)

def apply_filtering(diagrams: numpy.ndarray, eps: float):
    filtering = gtda.diagrams.Filtering(epsilon = eps)
    return filtering.fit_transform(diagrams)


def set_random_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


AMPLITUDE_METRICS = [
    { "id": "bottleneck", "metric": "bottleneck", "metric_params": { } },

    { "id": "wasserstein-1", "metric": "wasserstein", "metric_params": { "p": 1 } },
    { "id": "wasserstein-2", "metric": "wasserstein", "metric_params": { "p": 2 } },

    { "id": "betti-1", "metric": "betti", "metric_params": { "p": 1, 'n_bins': -1 } },
    { "id": "betti-2", "metric": "betti", "metric_params": { "p": 2, 'n_bins': -1 } },
    
    { "id": "landscape-1-1", "metric": "landscape", "metric_params": { "p": 1, "n_layers": 1, 'n_bins': -1 } },
    { "id": "landscape-1-2", "metric": "landscape", "metric_params": { "p": 1, "n_layers": 2, 'n_bins': -1 } },
    { "id": "landscape-2-1", "metric": "landscape", "metric_params": { "p": 2, "n_layers": 1, 'n_bins': -1 } },
    { "id": "landscape-2-2", "metric": "landscape", "metric_params": { "p": 2, "n_layers": 2, 'n_bins': -1 } },

    { "id": "silhouette-1-1", "metric": "silhouette", "metric_params": { "p": 1, "power": 1, 'n_bins': -1 } },
    { "id": "silhouette-1-2", "metric": "silhouette", "metric_params": { "p": 1, "power": 2, 'n_bins': -1 } },
    { "id": "silhouette-2-1", "metric": "silhouette", "metric_params": { "p": 2, "power": 1, 'n_bins': -1 } },
    { "id": "silhouette-2-2", "metric": "silhouette", "metric_params": { "p": 2, "power": 2, 'n_bins': -1 } }
]


class FeatureCalculator:
    def __init__(
        self,
        n_jobs: int = -1,
        reduced: bool = False,
        random_state: int = 42,
        filtering_percentile: int = 10
    ):
        self.n_jobs = n_jobs
        self.reduced = reduced
        self.random_state = random_state
        self.filtering_percentile = filtering_percentile


    def calc_stats(self, data: numpy.ndarray, prefix: str = "") -> pandas.DataFrame:
        assert len(data.shape) == 1
        if data.shape == (0,): data = numpy.array([ 0 ])

        if self.reduced:
            stats = numpy.array([
                numpy.max(data), numpy.mean(data), numpy.std(data), numpy.sum(data),
                numpy.linalg.norm(data, ord = 1), numpy.linalg.norm(data, ord = 2)
            ])
            names = [ "max", "mean", "std", "sum", "norm-1", "norm-2" ]
        else:
            stats = numpy.array([
                numpy.max(data), numpy.mean(data), numpy.std(data), numpy.sum(data),
                numpy.percentile(data, 25), numpy.median(data), numpy.percentile(data, 75),
                scipy.stats.kurtosis(data), scipy.stats.skew(data), numpy.linalg.norm(data, ord = 1), numpy.linalg.norm(data, ord = 2)
            ])
            names = [ "max", "mean", "std", "sum", "percentile-25", "median", "percentile-75", "kurtosis", "skew", "norm-1", "norm-2" ]

        return pandas.DataFrame([ numpy.nan_to_num(stats) ], columns = [ f"{prefix} {name}" for name in names ])

    def calc_batch_stats(self, data: numpy.ndarray, homology_dimensions: numpy.ndarray, prefix: str = "") -> pandas.DataFrame:
        def process_batch(batch: numpy.ndarray) -> pandas.DataFrame:
            features = [ ]
            for dim, vec in zip(homology_dimensions, batch):
                features.append(self.calc_stats(vec, prefix = f'{prefix} dim-{dim}'))
            return pandas.concat(features, axis = 1)

        features = joblib.Parallel(return_as = 'generator', n_jobs = self.n_jobs)(
            joblib.delayed(process_batch)(batch) for batch in data
        )
        return pandas.concat(
            tqdm.tqdm(features, total = len(data), desc = prefix),
            axis = 0
        )
    

    def calc_betti_features(self, diagrams: numpy.ndarray, prefix: str = "") -> pandas.DataFrame:
        if self.reduced:
            print('Skipping Betti features')
            return pandas.DataFrame()
        print('Calculating Betti features')
        betti_curve = gtda.diagrams.BettiCurve(n_bins = 100, n_jobs = self.n_jobs)
        betti_derivative = gtda.curves.Derivative()
        betti_curves = betti_curve.fit_transform(diagrams)
        betti_curves = betti_derivative.fit_transform(betti_curves)
        return self.calc_batch_stats(betti_curves, betti_curve.homology_dimensions_, f'{prefix} betti')

    def calc_landscape_features(self, diagrams: numpy.ndarray, prefix: str = "") -> pandas.DataFrame:
        print('Calculating landscape features')
        persistence_landscape = gtda.diagrams.PersistenceLandscape(n_layers = 1, n_bins = 100, n_jobs = self.n_jobs)
        landscape = persistence_landscape.fit_transform(diagrams)
        return self.calc_batch_stats(landscape, persistence_landscape.homology_dimensions_, f'{prefix} landscape')

    def calc_silhouette_features(self, diagrams: numpy.ndarray, prefix: str = "", powers: int = [ 1, 2 ]) -> pandas.DataFrame:
        if isinstance(powers, int):
            silhouette = gtda.diagrams.Silhouette(power = powers, n_bins = 100, n_jobs = self.n_jobs)
            silhouettes = silhouette.fit_transform(diagrams)
            return self.calc_batch_stats(silhouettes, silhouette.homology_dimensions_, f'{prefix} silhouette-{powers}')
        else:
            print('Calculating silhouette features')
            features = [ ]
            for power in powers:
                features.append(self.calc_silhouette_features(diagrams, prefix, power))
            return pandas.concat(features, axis = 1)
    

    def calc_entropy_features(self, diagrams: numpy.ndarray, prefix: str = "") -> pandas.DataFrame:
        print('Calculating entropy features')
        entropy = gtda.diagrams.PersistenceEntropy(normalize = True, nan_fill_value = 0, n_jobs = self.n_jobs)
        features = entropy.fit_transform(diagrams)
        names = [ f'{prefix} entropy dim-{dim}' for dim in entropy.homology_dimensions_ ]
        return pandas.DataFrame(features, columns = names)
    
    def calc_number_of_points_features(self, diagrams: numpy.ndarray, prefix: str = "") -> pandas.DataFrame:
        if self.reduced:
            print('Skipping number of points features')
            return pandas.DataFrame()
        print('Calculating number of points features')
        number_of_points = gtda.diagrams.NumberOfPoints(n_jobs = self.n_jobs)
        features = number_of_points.fit_transform(diagrams)
        names = [ f'{prefix} numberofpoints dim-{dim}' for dim in number_of_points.homology_dimensions_ ]
        return pandas.DataFrame(features, columns = names)
    
    def calc_amplitude_features(self, diagrams: numpy.ndarray, prefix: str = "", **metric) -> pandas.DataFrame:
        if len(metric) == 0:
            print('Calculating amplitude features')
            features = [ ]
            for metric in tqdm.tqdm(AMPLITUDE_METRICS, desc = f'{prefix} amplitudes'):
                if self.reduced and (metric['id'] == 'betti-1' or metric['id'] == 'betti-2'):
                    continue
                features.append(self.calc_amplitude_features(diagrams, prefix, **metric))
            return pandas.concat(features, axis = 1)
        else:
            metric_params = metric['metric_params'].copy()
            if metric_params.get('n_bins', None) == -1:
                metric_params['n_bins'] = 100
            amplitude = gtda.diagrams.Amplitude(metric = metric['metric'], metric_params = metric_params, n_jobs = self.n_jobs)
            features = amplitude.fit_transform(diagrams)
            return pandas.concat([
                pandas.DataFrame(features, columns = [ f'{prefix} amplitude-{metric["id"]} dim-{dim}' for dim in amplitude.homology_dimensions_ ]),
                pandas.DataFrame(numpy.linalg.norm(features, axis = 1, ord = 1).reshape(-1, 1), columns = [ f'{prefix} amplitude-{metric["id"]} norm-1' ]),
                pandas.DataFrame(numpy.linalg.norm(features, axis = 1, ord = 2).reshape(-1, 1), columns = [ f'{prefix} amplitude-{metric["id"]} norm-2' ])
            ], axis = 1)
    
    def calc_lifetime_features(self, diagrams: numpy.ndarray, prefix: str = "", eps: float = 0.0) -> pandas.DataFrame:
        if len(diagrams.shape) == 3:
            print('Calculating lifetime features')
            features = [ ]
            features = joblib.Parallel(return_as = 'generator', n_jobs = self.n_jobs)(
                joblib.delayed(self.calc_lifetime_features)(diag, prefix, eps)
                for diag in diagrams
            )
            return pandas.concat(
                tqdm.tqdm(features, total = len(diagrams), desc = f'{prefix} lifetime'),
                axis = 0
            )

        birth, death, dim = diagrams[:, 0], diagrams[:, 1], diagrams[:, 2]
        life = death - birth
        
        birth, death, dim = birth[life >= eps], death[life >= eps], dim[life >= eps]
        bd2 = (birth + death) / 2.0
        life = death - birth

        bd2_features = [ self.calc_stats(bd2, f'{prefix} bd2 all') ]
        life_features = [ self.calc_stats(life, f'{prefix} life all') ]
        for d in numpy.unique(diagrams[:, 2]).astype(int):
            bd2_features.append(self.calc_stats(bd2[dim == d], f'{prefix} bd2 dim-{d}'))
            life_features.append(self.calc_stats(life[dim == d], f'{prefix} life dim-{d}'))
        return pandas.concat([ *life_features, *bd2_features ], axis = 1)
        

    def calc_features(self, diagrams: numpy.ndarray, prefix: str = "") -> pandas.DataFrame:
        set_random_seed(self.random_state)
        
        eps = determine_filtering_epsilon(diagrams, self.filtering_percentile)
        diagrams = apply_filtering(diagrams, eps)
        print('Filtered diagrams:', diagrams.shape)
        return pandas.concat([
           self.calc_betti_features           (diagrams, prefix     ).reset_index(drop = True),
           self.calc_landscape_features       (diagrams, prefix     ).reset_index(drop = True),
           self.calc_silhouette_features      (diagrams, prefix     ).reset_index(drop = True),
           self.calc_entropy_features         (diagrams, prefix     ).reset_index(drop = True),
           self.calc_number_of_points_features(diagrams, prefix     ).reset_index(drop = True),
           self.calc_amplitude_features       (diagrams, prefix     ).reset_index(drop = True),
           self.calc_lifetime_features        (diagrams, prefix, eps).reset_index(drop = True)
        ], axis = 1)
