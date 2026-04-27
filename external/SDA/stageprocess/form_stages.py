import numpy

def form_stages(labels: numpy.ndarray) -> numpy.ndarray:
    edges = []
    n_clusters = len(numpy.unique(labels))
    for i in range(n_clusters):
        cl_samples = numpy.where(labels == i)[0]
        edges.append(cl_samples[0])
        edges.append(cl_samples[-1] + 1)
    return numpy.unique(edges)