import mne
import numpy
import pandas

from .. import stageprocess

def stage_timing(edges: numpy.ndarray, epochs: mne.Epochs) -> pandas.DataFrame:
    items = [ ]
    for (smin, smax) in stageprocess.form_stage_bands(edges)[0]:
        start = epochs.events[smin][0] / epochs.info['sfreq']
        end = epochs.events[smax][0] / epochs.info['sfreq']
        items.append({
            'Start-end time, sec': (start, end),
            'Time length, sec': end - start,
            'Number of epochs': smax - smin + 1
        })
    return pandas.DataFrame(items).transpose()