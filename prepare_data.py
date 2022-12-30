import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


def prepare_iris():
    iris = load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])

    return data

def prepare_vowel(filename):
    df = pd.read_csv(filename, delimiter=',',
                           names=['row.names', 'y', 'x.1', 'x.2', 'x.3', 'x.4', 'x.5', 'x.6', 'x.7', 'x.8', 'x.9',
                                  'x.10']).drop(columns='row.names')
    cols = df.columns.tolist()
    cols = cols[1:] + cols[:1] #move first column to last
    data = df[cols]

    return data


