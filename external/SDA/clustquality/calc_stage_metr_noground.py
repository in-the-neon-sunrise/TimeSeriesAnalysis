import numpy
import pandas
import typing

from .cluster_metrics_noground import cluster_metrics_noground, METRIC_NAMES

# Calculating clustering noground metrics for adjacent pairs of stages (Silh, Cal-Har, Dav-Bold)
def calc_stage_metr_noground(
    features: numpy.ndarray,
    st_edges: numpy.ndarray,
    metric_names: typing.List[str] = METRIC_NAMES
) -> pandas.DataFrame:
    metrics = [ ]
    for prev, cur, next in zip(st_edges[:-2], st_edges[1:-1], st_edges[2:]):
        labels = (numpy.arange(prev, next) >= cur).astype(numpy.int64)
        metrics.append(cluster_metrics_noground(features[prev:next], labels, metric_names))
    return pandas.DataFrame(metrics)
