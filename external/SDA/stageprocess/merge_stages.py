import enum
import typing

import numpy

from .merge_stages_1st_step import merge_stages_1st_step
from .merge_stages_2nd_step import merge_stages_2nd_step

class StageMerging(enum.Enum):
    NONE = 0
    FIRST = 1
    SECOND = 2
    BOTH = 3

def merge_stages_iter(
    features: numpy.ndarray,
    st_edges: numpy.ndarray,
    merging: StageMerging,
    len_threshold: int,
    dist_rate: float
) -> numpy.ndarray:
    if merging in [ StageMerging.FIRST, StageMerging.BOTH ]:
        st_edges = merge_stages_1st_step(features, st_edges, len_threshold)
    if merging in [ StageMerging.SECOND, StageMerging.BOTH ]:
        st_edges = merge_stages_2nd_step(features, st_edges, dist_rate)
    return st_edges

def merge_stages(
    features: numpy.ndarray,
    st_edges: numpy.ndarray,
    merging: StageMerging,
    len_thresholds: typing.List[int],
    dist_rate: float
) -> typing.Dict[int, numpy.ndarray]:
    if merging == StageMerging.NONE:
        return { None: st_edges }
    return {
        len_min: merge_stages_iter(features, st_edges, merging, len_min, dist_rate)
        for len_min in len_thresholds
    }
