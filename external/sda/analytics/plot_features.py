import math

import numpy
import pandas
import matplotlib.pyplot as plt

from .. import stageprocess

def plot_features(df_features: pandas.DataFrame, edges: numpy.ndarray) -> plt.Figure:
    stats = df_features.describe()
    num_features = df_features.shape[1]
    st_bands, _ = stageprocess.form_stage_bands(edges)
    st_centers = [ (smin + smax) / 2 for (smin, smax) in st_bands ]
    clust_stats = [ df_features[smin:(smax+1)].describe() for (smin, smax) in st_bands ]

    ncols = min(num_features, 4)
    nrows = math.ceil(num_features / 4)
    fig, axes = plt.subplots(nrows, ncols, figsize = (20, 3 * nrows))
    if num_features != 1: axes = axes.flat
    else: axes = [ axes ]
    
    for ax, feature in zip(axes, df_features):
        ax.set_title(feature)
        ax.plot(df_features[feature], color = 'blue')
        ax.tick_params(axis = 'both', labelsize = 8, direction = 'in')
        ax.vlines(x = edges[1:-1], ymin = stats[feature]['min'], ymax = stats[feature]['max'], color = 'black', linewidth = 1.8)
    
        # Add stage index 
        for idx, center in enumerate(st_centers):
            ax.text(center, stats[feature]['max'], idx + 1, fontsize = 9, color = 'black', horizontalalignment = 'center', fontweight = 'bold')

        # Error bars
        y_min = numpy.array([ clust_stat[feature]['25%'] for clust_stat in clust_stats ])
        y_avg = numpy.array([ clust_stat[feature]['50%'] for clust_stat in clust_stats ])
        y_max = numpy.array([ clust_stat[feature]['75%'] for clust_stat in clust_stats ])
        ax.errorbar(st_centers, y_avg, [ y_avg - y_min, y_max - y_avg ], capsize = 4, linewidth = 2.5, color = 'red')

    return fig