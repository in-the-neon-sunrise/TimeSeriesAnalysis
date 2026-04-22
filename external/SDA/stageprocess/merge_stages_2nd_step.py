import numpy

from .calc_stage_distances import calc_stage_distances_ward

def merge_stages_2nd_step_iter(features: numpy.ndarray, st_edges: numpy.ndarray) -> numpy.ndarray:
    return numpy.delete(st_edges, calc_stage_distances_ward(features, st_edges).argmin() + 1)

def should_stop(features: numpy.ndarray, st_edges: numpy.ndarray, dist_threshold: float) -> bool:
    st_dist_list = calc_stage_distances_ward(features, st_edges)
    return len(st_edges) <= 3 or st_dist_list.min() > dist_threshold * numpy.mean(st_dist_list)

# Merge stages if length > n_stages
def merge_stages_2nd_step(features: numpy.ndarray, st_edges: numpy.ndarray, dist_threshold: float) -> numpy.ndarray:
    while not should_stop(features, st_edges, dist_threshold):
        st_edges = merge_stages_2nd_step_iter(features, st_edges)
    return st_edges
