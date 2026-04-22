import typing
import itertools

import tqdm
import numpy
import pandas
import joblib
import scipy.sparse
import sklearn.cluster
import tqdm.contrib.itertools

from .form_stages import form_stages
from ..clustquality import cluster_metrics_noground
from .merge_stages import merge_stages, StageMerging

def stage1_iter(
    features: numpy.ndarray,

    n_clusters: int,
    k_neighbours: int,

    merging: StageMerging,
    len_thresholds: typing.List[int],
    dist_rate: float,

    calc_quality: bool
) -> pandas.DataFrame:
    # Prepare kwargs for Ward's clustering
    n_samples, _ = features.shape
    diag_nums = numpy.arange(-k_neighbours, k_neighbours + 1)
    diag_values = numpy.ones_like(diag_nums)
    connectivity = scipy.sparse.diags(diag_values, diag_nums, (n_samples, n_samples), 'csr', numpy.int8)

    # Ward's clustering
    kwargs = { 'n_clusters': n_clusters, 'linkage': 'ward', 'connectivity': connectivity }
    labels = sklearn.cluster.AgglomerativeClustering(**kwargs).fit_predict(features)
    
    # Calculate quality metrics
    metrics = cluster_metrics_noground(features, labels) if calc_quality else { }

    # Merge stages
    edges_lists = merge_stages(features, form_stages(labels), merging, len_thresholds, dist_rate)

    # Construct result
    return [
        { 'N_clusters': n_clusters, 'K_neighb': k_neighbours, 'Len_min': len_min, 'St_edges': edges, **metrics }
        for len_min, edges in edges_lists.items()
    ]

def stage1(
    features: numpy.ndarray,

    n_clusters: typing.List[int],
    k_neighbours: typing.List[int],

    merging: StageMerging,
    len_thresholds: typing.List[int],
    dist_rate: float,

    n_jobs: int,
    verbose: bool,
    calc_quality: bool
) -> pandas.DataFrame:
    loop = list(itertools.product(n_clusters, k_neighbours))
    df_st_edges = joblib.Parallel(return_as = 'generator', n_jobs = n_jobs)(
        joblib.delayed(stage1_iter)(
            features,
            n_clust, k_neighb,
            merging, len_thresholds, dist_rate,
            calc_quality
        ) for n_clust, k_neighb in loop
    )

    if verbose:
        df_st_edges = tqdm.tqdm(df_st_edges, total = len(loop), desc = 'stage 1')
    return pandas.DataFrame(list(itertools.chain(*df_st_edges)))
