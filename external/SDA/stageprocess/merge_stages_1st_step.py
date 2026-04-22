import numpy

from .clusters_dist_ward import clusters_dist_ward

def merge_stages_1st_step_iter(features: numpy.ndarray, st_edges: numpy.ndarray) -> numpy.ndarray:
    ind = (st_edges[1:] - st_edges[:-1]).argmin()

    if (ind == len(st_edges) - 2):
        return numpy.delete(st_edges, ind)

    if (ind == 0):
        return numpy.delete(st_edges, 1)

    clust_mid = features[st_edges[ind]:st_edges[ind + 1]]
    clust_left = features[st_edges[ind - 1]:st_edges[ind]]
    clust_right = features[st_edges[ind + 1]:st_edges[ind + 2]]
    
    st_dist_left = clusters_dist_ward(clust_left, clust_mid)
    st_dist_right = clusters_dist_ward(clust_mid, clust_right)
    return numpy.delete(st_edges, ind + (st_dist_left > st_dist_right))

def should_stop(st_edges: numpy.ndarray, len_threshold: int) -> bool:
    return len(st_edges) <= 3 or (st_edges[1:] - st_edges[:-1]).min() > len_threshold

# Merge small stages with neighbours
def merge_stages_1st_step(features: numpy.ndarray, st_edges: numpy.ndarray, len_threshold: int) -> numpy.ndarray:
    while not should_stop(st_edges, len_threshold):
        st_edges = merge_stages_1st_step_iter(features, st_edges)
    return st_edges
