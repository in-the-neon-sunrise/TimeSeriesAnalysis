import numpy
import pandas

def form_edges_all(df_st_edges: pandas.DataFrame, st_len: int, k_nb_max: int, n_cl: int) -> numpy.ndarray:
    len_min_mask = (df_st_edges['Len_min'] == st_len)
    k_neighb_mask = (df_st_edges['K_neighb'] <= k_nb_max)
    n_clusters_mask = (df_st_edges['N_clusters'] <= n_cl)
    mask = len_min_mask & k_neighb_mask & n_clusters_mask
    edges_list = df_st_edges[mask]['St_edges'].tolist()
    edges_all = [ edge for edges in edges_list for edge in edges[1:-1] ]
    return numpy.sort(edges_all).reshape(-1, 1)
