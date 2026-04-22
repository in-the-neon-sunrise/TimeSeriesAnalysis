import pandas

def best_results(result: pandas.DataFrame, key: str, min_stage_length: int = 0) -> pandas.DataFrame:
    best_results = [ ]
    n_st_edge_clusters_min = result['N_stages'].to_numpy().min()
    n_st_edge_clusters_max = result['N_stages'].to_numpy().max()
    for n_st_edge_clusters in range(n_st_edge_clusters_min, n_st_edge_clusters_max + 1):
        n_st_mask = result['N_stages'] == n_st_edge_clusters
        st_len_min_mask = result['St_len_min'] >= min_stage_length
        ok_rows = result[n_st_mask & st_len_min_mask].reset_index(drop = True)
        if (len(ok_rows) == 0):
            continue
        best_results.append(ok_rows.iloc[ok_rows[key].idxmax()].to_dict())
    return pandas.DataFrame(best_results)