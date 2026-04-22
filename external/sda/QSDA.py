import os
import enum
import typing

import tqdm
import numpy
import joblib
import pandas
import sklearn.preprocessing

from .SDA import SDA
from .SDA import StageMerging
from SDA.clustquality import calc_stage_metr_noground
from SDA.stageprocess import merge_stages_1st_step_iter, merge_stages_2nd_step_iter


class ScoreFunction(enum.Enum):
    ZERO = 0
    LOG = 1
    SQRT = 2
    MULTIPLE = 3
    SQUARE = 4


class QSDA(SDA):
    def __init__(
        self,
        n_jobs: int = 1,
        scale: bool = False,
        random_state: int = 42,

        qsda_n_jobs: int = -1,
        score_function: ScoreFunction = ScoreFunction.MULTIPLE,
        scores_folder: typing.Optional[str] = None,
        
        threshold: typing.Union[int, float] = 2 / 3,
        min_unique_values: int = 40,

        # stage 1
        n_clusters_min: typing.Optional[int] = 2, n_clusters_max: typing.Optional[int] = 15, n_clusters: typing.Optional[typing.Iterable[int]] = None,
        k_neighbours_min: typing.Optional[int] = None, k_neighbours_max: typing.Optional[int] = None, k_neighbours: typing.Optional[typing.Iterable[int]] = range(20, 51, 5),
        
        # stage 1 - merging
        st1_merging: StageMerging = StageMerging.BOTH,
        st1_len_thresholds: typing.Union[int, typing.Iterable[int]] = 40,
        st1_dist_rate: float = 0.3,

        # stage 2
        n_cl_max_thr: typing.Iterable[int] = [20],
        k_neighb_max_thr: typing.Iterable[int] = [50],
        n_edge_clusters_min: int = 2, n_edge_clusters_max: int = 10, n_edge_clusters: typing.Optional[typing.Iterable[int]] = None,
        
        # stage 2 - merging
        st2_merging: StageMerging = StageMerging.BOTH,
        st2_len_thresholds: typing.Union[int, typing.Iterable[int]] = 40,
        st2_dist_rate: float = 0.2
    ):
        super().__init__(
            n_jobs = n_jobs, scale = scale, verbose = False, random_state = random_state,

            st1_calc_quality = False,
            n_clusters_min = n_clusters_min, n_clusters_max = n_clusters_max, n_clusters = n_clusters,
            k_neighbours_min = k_neighbours_min, k_neighbours_max = k_neighbours_max, k_neighbours = k_neighbours,
            st1_merging = st1_merging, st1_len_thresholds = st1_len_thresholds, st1_dist_rate = st1_dist_rate,

            st2_calc_quality = False, n_cl_max_thr = n_cl_max_thr, k_neighb_max_thr = k_neighb_max_thr,
            n_edge_clusters_min = n_edge_clusters_min, n_edge_clusters_max = n_edge_clusters_max, n_edge_clusters = n_edge_clusters,
            st2_merging = st2_merging, st2_len_thresholds = st2_len_thresholds, st2_dist_rate = st2_dist_rate
        )

        self.qsda_n_jobs = qsda_n_jobs
        self.score_function = score_function
        self.scores_folder = scores_folder

        self.threshold = threshold
        self.min_unique_values = min_unique_values


    def generate_merges(self, feature: numpy.ndarray, st_edges: numpy.ndarray) -> typing.Set[typing.Tuple[int]]:
        if len(st_edges) <= 3: return set([ tuple(st_edges) ])
        edges1 = merge_stages_1st_step_iter(feature, st_edges)
        edges2 = merge_stages_2nd_step_iter(feature, st_edges)
        merges1 = self.generate_merges(feature, edges1)
        merges2 = self.generate_merges(feature, edges2)
        return set([ tuple(st_edges) ]) | merges1 | merges2

    def calc_score(self, feature: numpy.ndarray, st_edges: numpy.ndarray) -> float:
        metrics = calc_stage_metr_noground(feature, st_edges, [ 'Silh' ]).mean()
        match self.score_function:
            case ScoreFunction.ZERO:
                return metrics['Silh']
            case ScoreFunction.LOG:
                return numpy.log2(len(st_edges)) * metrics['Silh']
            case ScoreFunction.SQRT:
                return numpy.sqrt(len(st_edges)) * metrics['Silh']
            case ScoreFunction.MULTIPLE:
                return len(st_edges) * metrics['Silh']
            case ScoreFunction.SQUARE:
                return len(st_edges) * len(st_edges) * metrics['Silh']
        raise NotImplementedError(function)


    def get_feature_score(self, name: str, feature: numpy.ndarray, edges_lists: typing.List[numpy.ndarray]) -> dict:
        scores_file = f"{self.scores_folder}/{name}/scores.npy"
        if self.scores_folder is not None and os.path.exists(scores_file):
            scores = numpy.load(scores_file)
        else:
            scores = [ self.calc_score(feature, merge) for merge in edges_lists ]
            numpy.save(scores_file, scores)

        return {
            'name': name,
            'score': numpy.max(scores),
            'mean': feature.mean(),
            'variance': feature.var(),
            'unique_values': len(numpy.unique(feature))
        }
    
    def get_edges_lists(self, name: str, feature: numpy.ndarray, results: pandas.DataFrame) -> typing.List[numpy.ndarray]:
        edges_lists_file = f"{self.scores_folder}/{name}/edges_lists.txt"
        if self.scores_folder is not None and os.path.exists(edges_lists_file):
            edges_lists = [ ]
            for line in open(edges_lists_file, 'r'):
                edges_list = numpy.fromstring(line.strip('[]'), sep = ',', dtype = numpy.uint)
                edges_lists.append(edges_list)
            return edges_lists
        
        edges_lists = set()
        for edges in map(numpy.array, set(map(tuple, results["St_edges"]))):
            edges_lists |= self.generate_merges(feature, edges)
        edges_lists = list(map(numpy.array, edges_lists))
        
        file = open(edges_lists_file, 'w')
        for edges_list in edges_lists:
            file.write(f"[{','.join(map(str, edges_list))}]\n")

        return edges_lists
    
    def get_results(self, name: str, feature: numpy.ndarray) -> pandas.DataFrame:
        results_file = f"{self.scores_folder}/{name}/results.csv"
        if self.scores_folder is not None and os.path.exists(results_file):
            results = pandas.read_csv(results_file)
            st_edges_list = [ ]
            for st_edges in results["St_edges"]:
                st_edges = numpy.fromstring(st_edges.strip('[]'), sep = ' ', dtype = numpy.uint)
                st_edges_list.append(st_edges)
            results["St_edges"] = st_edges_list
            return results
        
        results, _ = self.apply(feature)
        results.to_csv(results_file, index = False)
        return results

    def score_feature(self, name: str, feature: numpy.ndarray):
        feature = feature.reshape(-1, 1)
        
        folder = f"{self.scores_folder}/{name}"
        if self.scores_folder is not None:
            os.makedirs(folder, exist_ok = True)

        scores_file = f"{folder}/scores.npy"
        edges_lists_file = f"{folder}/edges_lists.npy"
        
        if self.scores_folder is not None and os.path.exists(scores_file):
            return self.get_feature_score(name, feature, None)
        
        if self.scores_folder is not None and os.path.exists(edges_lists_file):
            edges_lists = self.get_edges_lists(name, feature, None)
            return self.get_feature_score(name, feature, edges_lists)
        
        results = self.get_results(name, feature)
        edges_lists = self.get_edges_lists(name, feature, results)
        return self.get_feature_score(name, feature, edges_lists)


    def select(self, features: pandas.DataFrame) -> typing.Tuple[pandas.DataFrame, pandas.DataFrame]:
        scores_file = f"{self.scores_folder}/scores.csv"

        if self.scores_folder is not None and os.path.exists(scores_file):
            scores = pandas.read_csv(scores_file)
        else:
            scores = joblib.Parallel(return_as = 'generator', n_jobs = self.qsda_n_jobs)(
                joblib.delayed(self.score_feature)(name, feature.to_numpy())
                for name, feature in features.items()
            )
            scores = pandas.DataFrame(list(tqdm.tqdm(scores, total = features.shape[1], desc = 'scores')))
            scores["normalized_score"] = sklearn.preprocessing.MinMaxScaler().fit_transform(scores[["score"]])
            scores.to_csv(scores_file, index = False)

        if isinstance(self.threshold, int):
            score_values = scores["normalized_score"].to_numpy()
            score_values = numpy.sort(score_values)[::-1]
            threshold = numpy.round(score_values[self.threshold], 2)
        else:
            threshold = self.threshold
        print('Using threshold', threshold)
        
        score_filter = scores["normalized_score"] >= threshold
        unique_values_filter = scores["unique_values"] >= self.min_unique_values

        feature_idx = list(scores[score_filter & unique_values_filter]["name"])
        return features[feature_idx], scores
