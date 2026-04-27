import typing

import numpy

# Form bands array and new_labels list for stages
def form_stage_bands(st_edges: numpy.ndarray) -> typing.Tuple[typing.List[typing.Tuple[int, int]], numpy.ndarray]:
    st_bands = [ ]
    new_labels = numpy.empty(st_edges[-1])
    for i, (cur, next) in enumerate(zip(st_edges[:-1], st_edges[1:])):
        st_bands.append((cur, next - 1))
        new_labels[cur:next] = i
    return st_bands, new_labels