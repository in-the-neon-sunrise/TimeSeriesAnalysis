import tqdm
import numpy
import pandas
import joblib

from .calc_IV import calc_IV

def calc_IV_feature(feature_name: str, feature: numpy.ndarray, labels: numpy.ndarray, bins: int = 10) -> dict:
    IVs = [ ]
    for cluster in numpy.unique(labels):
        target = (labels == cluster).astype(int)
        IVs.append(calc_IV(feature, target, bins))
    return { 'Feature': feature_name, 'IV': numpy.mean(IVs), 'IVs': IVs }

# Calculation of IV and WoE for given DataFrame of features and given clustering labels 
def calc_IV_clust(features: pandas.DataFrame, labels: numpy.ndarray, bins: int = 10, n_jobs: int = -1) -> pandas.DataFrame:
    IV = joblib.Parallel(return_as = 'generator', n_jobs = n_jobs)(
        joblib.delayed(calc_IV_feature)(feature_name, features[feature_name].to_numpy(), labels, bins)
        for feature_name in features.columns
    )
    return pandas.DataFrame(list(tqdm.tqdm(IV, total = features.shape[1], desc = 'IV')))
