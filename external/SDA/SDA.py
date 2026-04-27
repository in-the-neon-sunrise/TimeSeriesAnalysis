import typing

import numpy
import pandas
import sklearn.preprocessing

from .stageprocess import StageMerging, stage1, stage2

def to_list(item: typing.Union[int, typing.List[int]]) -> typing.List[int]:
    return item if isinstance(item, list) else [ item ]

class SDA:
    def __init__(
        self,
        n_jobs: int = -1,
        scale: bool = False,
        verbose: bool = True,
        random_state: int = 42,

        # stage 1
        st1_calc_quality: bool = True,
        n_clusters_min: typing.Optional[int] = 2, n_clusters_max: typing.Optional[int] = 20, n_clusters: typing.Optional[typing.Iterable[int]] = None,
        k_neighbours_min: typing.Optional[int] = 20, k_neighbours_max: typing.Optional[int] = 50, k_neighbours: typing.Optional[typing.Iterable[int]] = None,
        
        # stage 1 - merging
        st1_merging: StageMerging = StageMerging.BOTH,
        st1_len_thresholds: typing.Union[int, typing.Iterable[int]] = [0, 20, 40, 60],
        st1_dist_rate: float = 0.3,

        # stage 2
        st2_calc_quality: bool = True,
        n_cl_max_thr: typing.Iterable[int] = [10, 15, 20],
        k_neighb_max_thr: typing.Iterable[int] = [35, 40, 45, 50],
        n_edge_clusters_min: int = 2, n_edge_clusters_max: int = 15, n_edge_clusters: typing.Optional[typing.Iterable[int]] = None,
        
        # stage 2 - merging
        st2_merging: StageMerging = StageMerging.BOTH,
        st2_len_thresholds: typing.Union[int, typing.Iterable[int]] = [40],
        st2_dist_rate: float = 0.2
    ):
        self.scale = scale
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        self.st1_calc_quality = st1_calc_quality
        self.n_clusters = n_clusters or range(n_clusters_min, n_clusters_max + 1)
        self.k_neighbours = k_neighbours or range(k_neighbours_min, k_neighbours_max + 1)
        
        self.st1_merging = st1_merging
        self.st1_len_thresholds = to_list(st1_len_thresholds)
        self.st1_dist_rate = st1_dist_rate
        
        self.st2_calc_quality = st2_calc_quality
        self.n_cl_max_thr = n_cl_max_thr
        self.k_neighb_max_thr = k_neighb_max_thr
        self.n_edge_clusters = n_edge_clusters or range(n_edge_clusters_min, n_edge_clusters_max + 1)
        
        self.st2_merging = st2_merging
        self.st2_len_thresholds = to_list(st2_len_thresholds)
        self.st2_dist_rate = st2_dist_rate

    def apply(
        self,
        features: numpy.ndarray,
        df_st_edges: typing.Optional[pandas.DataFrame] = None
    ) -> typing.Tuple[pandas.DataFrame, pandas.DataFrame]:
        if self.scale:
            features = sklearn.preprocessing.StandardScaler().fit_transform(features)

        if self.verbose:
            print('Applying to {} samples with {} features each'.format(*features.shape))

        if df_st_edges is None:
            df_st_edges = stage1(
                features,
                self.n_clusters, self.k_neighbours,
                self.st1_merging, self.st1_len_thresholds, self.st1_dist_rate,
                self.n_jobs, self.verbose, self.st1_calc_quality
            )

        result = stage2(
            features, df_st_edges,
            self.k_neighb_max_thr, self.n_cl_max_thr, self.n_edge_clusters,
            self.st2_merging, self.st2_len_thresholds, self.st2_dist_rate,
            self.n_jobs, self.verbose, self.random_state, self.st2_calc_quality
        )
        
        return result, df_st_edges
