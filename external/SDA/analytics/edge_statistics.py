import numpy
import pandas

from external.SDA import clustquality, stageprocess


def edge_statistics(features: numpy.ndarray, st_edges: numpy.ndarray) -> pandas.DataFrame:
    return {
        **clustquality.calc_stage_metr_noground(features, st_edges),
        'Ward': stageprocess.calc_stage_distances_ward(features, st_edges),
        'Centr': stageprocess.calc_stage_distances_centroid(features, st_edges)
    }
