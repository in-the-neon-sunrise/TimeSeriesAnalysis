import numpy

def clusters_dist_ward(cluster1: numpy.ndarray, cluster2: numpy.ndarray) -> float:
    n1 = len(cluster1)
    n2 = len(cluster2)
    cl1_center = cluster1.mean(axis = 0)
    cl2_center = cluster2.mean(axis = 0)
    return (n1 * n2 / (n1 + n2)) * numpy.linalg.norm(cl1_center - cl2_center) ** 2