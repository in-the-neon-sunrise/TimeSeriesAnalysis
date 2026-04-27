import typing

import numpy
import sklearn.metrics

# Define the metrics which require only data and predicted labels
cluster_metrics = [
    (sklearn.metrics.silhouette_score, 'Silh'),
    (sklearn.metrics.calinski_harabasz_score, 'Cal-Har'),
    (sklearn.metrics.davies_bouldin_score, 'Dav-Bold')
]

METRIC_NAMES = [ metric[1] for metric in cluster_metrics ]

def cluster_metrics_noground(
    data: numpy.ndarray,
    labels_pred: numpy.ndarray,
    metric_names: typing.List[str] = METRIC_NAMES
) -> typing.Dict[str, float]:
    if len(numpy.unique(labels_pred)) <= 1:
        return {
            name: 0.0
            for (_, name) in cluster_metrics
            if name in metric_names
        }

    return {
        name: func(data, labels_pred)
        for (func, name) in cluster_metrics
        if name in metric_names
    }
