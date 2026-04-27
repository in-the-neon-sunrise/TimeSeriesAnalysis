import mne
import numpy
import pandas
import sklearn.cluster
import matplotlib.pyplot as plt
import matplotlib.patches as ptchs

from .. import stageprocess
from .stage_timing import stage_timing
from .edge_statistics import edge_statistics

def plot_stats(features: numpy.ndarray, epochs: mne.Epochs, result: dict, df_st_edges: pandas.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize = (7.5, 4))
    edges = numpy.array(result['St_edges'])

    ymin, ymax = -0.2, 1.225
    ax.set_ylim(ymin, ymax)
    ax.grid(axis = 'y')
    ax.yaxis.tick_right()
    ax.set_xlim(0, len(features))
    ax.xaxis.set_visible(False)
    ax.vlines(edges[1:-1], ymin, ymax, color = 'black', linewidth = 1) # Stage boundaries

    st_time_len = stage_timing(edges, epochs).iloc[1]
    st_bands, _ = stageprocess.form_stage_bands(edges)
    for i, ((smin, smax), length) in enumerate(zip(st_bands, st_time_len)):
        center = (smin + smax) / 2
        color = plt.get_cmap('Set3')(i)
        ax.axvspan(smin, smax, alpha = 0.3, color = color) # Background color
        ax.text(center, ymax - 0.15, i + 1, fontsize = 10, fontweight = 'bold', horizontalalignment = 'center') # Name
        ax.text(center, -0.13, '{}s'.format(round(length)), fontsize = 9, fontstyle = 'italic', horizontalalignment = 'center') # Length
        ax.add_patch(ptchs.Rectangle((smin, -0.20), smax - smin, 0.05, edgecolor = 'black', facecolor = color, fill = True, lw = 1)) # Stage

    stats = edge_statistics(features, edges)
    for idx, column in enumerate(stats):
        color = plt.get_cmap('Set1')(idx)
        stats[column] /= stats[column].max()
        ax.plot(edges[1:-1], stats[column], linestyle = '--', marker = 'o', color = color, label = column)
    
    st_edges_all = stageprocess.form_edges_all(df_st_edges, result['St_len_min'], result['K_nb_max'], result['N_cl_max'])
    kwargs = { 'n_clusters': result['N_stages'] - 1, 'random_state': 0, 'n_init': 10 }
    labels = sklearn.cluster.KMeans(**kwargs).fit_predict(st_edges_all)
    for i, label in enumerate(sorted(set(labels), key = list(labels).index)):
        color = plt.get_cmap('Set3')(i)
        x_sc = [ e for e, l in zip(st_edges_all, labels) if l == label ]
        s = [ 1.5 * list(x_sc).count(x) for x in x_sc ]
        ax.scatter(x_sc, numpy.full_like(x_sc, 0.0), s = s, color = color)

    ax.legend(loc = 'lower center', ncols = 5, bbox_to_anchor = (0.5, 0.933), framealpha = 1.0)
    return fig