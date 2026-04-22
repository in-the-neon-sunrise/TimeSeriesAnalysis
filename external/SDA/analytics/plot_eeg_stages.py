import mne
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as ptchs

from .stage_timing import stage_timing

def plot_eeg_stages(data: mne.io.Raw, edges: numpy.ndarray, epochs: mne.Epochs, resolution: float = 1) -> plt.Figure:
    events = mne.make_fixed_length_events(data, duration = resolution)
    draw_epochs = mne.Epochs(data, events, baseline = None, tmin = 0, tmax = resolution, preload = True, verbose = False)
    data = numpy.average(draw_epochs.get_data(copy = True), axis = 2)
    data = data[:, data.mean(axis = 0).argsort()[::-1]] # Sort for better picture
    min, max = numpy.min(data), numpy.max(data)

    edges_sec = [ ]
    for edge in edges:
        if edge >= len(epochs.events): edge -= 1
        edges_sec.append(epochs.events[edge][0] / epochs.info['sfreq'])
    st_time_len = stage_timing(edges, epochs).iloc[1]

    fig, ax = plt.subplots(1, 1, figsize = (12.5, 4))
    ax.xaxis.set_visible(False)
    ax.set_ylim(min - 3e-6, max)

    for idx, (start, end, length) in enumerate(zip(edges_sec[:-1], edges_sec[1:], st_time_len)):
        center = (start + end) / 2
        color = plt.get_cmap('Set3')(idx)
        ax.axvspan(start, end, alpha = 0.3, color = color) # Background color
        ax.text(center, min + 3e-7, '{}s'.format(round(length)), fontsize = 9, fontstyle = 'italic', horizontalalignment = 'center') # Length
        ax.add_patch(ptchs.Rectangle((start, min - 3e-6), end - start, 2e-6, edgecolor = 'black', facecolor = color, fill = True, lw = 1)) # Stage
        ax.text(center, max - 4e-6, idx + 1, fontsize = 10, color = 'black', horizontalalignment = 'center', fontweight = 'bold') # Index

    x = draw_epochs.events[:, 0] / draw_epochs.info['sfreq']
    ax.set_xlim(x[0], x[-1])
    for i in range(0, data.shape[1]): ax.plot(x, data[:, i])
    ax.vlines(x = edges_sec, ymin = min - 3e-6, ymax = max, color = 'black', linewidth = 1.8)

    return fig