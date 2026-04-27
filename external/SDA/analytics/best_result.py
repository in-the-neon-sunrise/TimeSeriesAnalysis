import pandas

def best_result(result: pandas.DataFrame, key: str, n_stages: int, min_stage_length: int = 0) -> dict:
    n_st_mask = result['N_stages'] == n_stages
    st_len_min_mask = result['St_len_min'] >= min_stage_length
    ok_rows = result[n_st_mask & st_len_min_mask].reset_index(drop = True)
    return ok_rows.iloc[ok_rows[key].idxmax()].to_dict()