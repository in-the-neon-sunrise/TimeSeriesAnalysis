import numpy
import pandas

def calc_IV(feature: numpy.ndarray, target: numpy.ndarray, bins: int = 10) -> float:
    if len(numpy.unique(feature)) > 10:
        feature = pandas.qcut(feature, bins, duplicates = 'drop')
    df = pandas.DataFrame({ 'x': feature, 'y': target })
    df = df.groupby("x", as_index = False, observed = False).agg({ "y": [ "count", "sum" ] })
    df.columns = [ 'Cutoff', 'N', 'Events' ]

    # Calculate % of events in each group.
    df['% of Events'] = numpy.maximum(df['Events'], 0.5) / df['Events'].sum()

    # Calculate the non events in each group.
    df['Non-Events'] = df['N'] - df['Events']
    # Calculate % of non events in each group.
    df['% of Non-Events'] = numpy.maximum(df['Non-Events'], 0.5) / df['Non-Events'].sum()

    # Calculate WOE by taking natural log of division of % of non-events and % of events
    df['WoE'] = numpy.log(df['% of Events'] / df['% of Non-Events'])
    df['IV'] = df['WoE'] * (df['% of Events'] - df['% of Non-Events'])

    return df['IV'].sum()