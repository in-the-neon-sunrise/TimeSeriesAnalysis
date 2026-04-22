import typing

import numpy

from .clusters_dist_ward import clusters_dist_ward

# Calculating stage distances (Ward)
def calc_stage_distances_ward(features: numpy.ndarray, st_edges: numpy.ndarray) -> numpy.ndarray:
    return numpy.array([
        clusters_dist_ward(
            features[st_edges[i - 1]:st_edges[i]],
            features[st_edges[i]:st_edges[i + 1]]
        ) for i in range(1, len(st_edges) - 1)
    ])
    
# Calculating stage distances (Centroid)
def calc_stage_distances_centroid(features: numpy.ndarray, st_edges: numpy.ndarray) -> numpy.ndarray:
    return numpy.array([
        numpy.linalg.norm(
            features[st_edges[i - 1]:st_edges[i]].mean(axis = 0)
            -
            features[st_edges[i]:st_edges[i + 1]].mean(axis = 0)
        ) for i in range(1, len(st_edges) - 1)
    ])

