import typing
import itertools

import numpy
import pandas
import joblib
import scipy.stats
import sklearn.cluster
import tqdm.contrib.itertools

from .form_edges_all import form_edges_all
from .form_stage_bands import form_stage_bands
from .merge_stages import merge_stages, StageMerging
from .calc_stage_distances import calc_stage_distances_ward
from .calc_stage_distances import calc_stage_distances_centroid
from external.SDA.clustquality import cluster_metrics_noground, calc_stage_metr_noground

def stage2_iter(
    features: numpy.ndarray,
    df_st_edges: pandas.DataFrame,

    st_len: int,
    k_nb_max: int,
    n_cl: int,
    n_edge_clusters: int,

    merging: StageMerging,
    len_thresholds: typing.Union[int, typing.List[int]],
    dist_rate: float,

    random_state: int,
    calc_quality: bool
) -> pandas.DataFrame:
    # Clustering stage edges
    st_edges_all = form_edges_all(df_st_edges, st_len, k_nb_max, n_cl)
    kwargs = { 'n_clusters': n_edge_clusters, 'random_state': random_state, 'n_init': 10 }
    labels = sklearn.cluster.KMeans(**kwargs).fit_predict(st_edges_all)

    # Form stages by centers of clusters (median, mean, mode)
    st_medians, st_modes, st_means = [], [], []
    for _st in range(n_edge_clusters):
        st_cluster = st_edges_all[numpy.where(labels == _st)[0]]
        if len(st_cluster) == 0:
            continue
        st_modes.append(int(scipy.stats.mode(st_cluster, nan_policy = 'omit').mode))
        st_medians.append(int(numpy.median(st_cluster)))
        st_means.append(int(numpy.mean(st_cluster)))

    result = [ ]
    for (st_centers, cl_center_type) in zip((st_medians, st_modes, st_means), ('Median', 'Mode')):
        # Calculate and merge stages
        st_edges_centers = numpy.array([ 0 ] + sorted(st_centers) + [ features.shape[0] ])
        edges_lists = merge_stages(features, st_edges_centers, merging, len_thresholds, dist_rate)

        # Construct result
        for len_min, edges in edges_lists.items():
            # Calculate quality metrics
            metrics = { }
            stage_lengths = edges[1:] - edges[:-1]
            
            if stage_lengths.min() > len_min:
                if calc_quality:
                
                    metrics = {
                        'N_stages': len(stage_lengths),
                        'Longest_stage': numpy.max(stage_lengths),
                        'Shortest_stage': numpy.min(stage_lengths),
                        'Avg_stage_length': numpy.mean(stage_lengths),
                        'Ward_dist': numpy.mean(calc_stage_distances_ward(features, edges)),
                        'Cen_dist': numpy.mean(calc_stage_distances_centroid(features, edges)),
                        **cluster_metrics_noground(features, form_stage_bands(edges)[1]),
                        **calc_stage_metr_noground(features, edges).mean().rename(lambda col: f'Avg-{col}').to_dict()
                    }

                result.append({
                    'St_len_min': st_len, 'K_nb_max': k_nb_max, 'N_cl_max': n_cl,
                    'Cl_cen': cl_center_type, 'Len_min': len_min, 'St_edges': edges,
                    **metrics
                })
    return result

def stage2(
    features: numpy.ndarray,
    df_st_edges: pandas.DataFrame,
    
    k_neighb_max_thr: typing.List[int],
    n_cl_max_thr: typing.List[int],
    n_edge_clusters: typing.List[int],

    merging: StageMerging,
    len_thresholds: typing.List[int],
    dist_rate: float,
    
    n_jobs: int,
    verbose: bool,
    random_state: int,
    calc_quality: bool
) -> pandas.DataFrame:
    st1_len_thresholds = set(df_st_edges['Len_min'])

    loop = list(itertools.product(st1_len_thresholds, k_neighb_max_thr, n_cl_max_thr, n_edge_clusters))
    result = joblib.Parallel(return_as = 'generator', n_jobs = n_jobs)(
        joblib.delayed(stage2_iter)(
            features, df_st_edges,
            st_len, k_nb_max, n_cl, n_edge_clusters,
            merging, len_thresholds, dist_rate,
            random_state, calc_quality
        ) for st_len, k_nb_max, n_cl, n_edge_clusters in loop
    )
    
    if verbose:
        result = tqdm.tqdm(result, total = len(loop), desc = 'stage 2')
    return pandas.DataFrame(list(itertools.chain(*result)))
