""" This file is the copy of ipynb file to follow the specification. 
This file is not yet very readable. Please refer pset1.ipynb """
# TODO: refactoring

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.model_selection
import sklearn.neural_network
from sklearn.model_selection import cross_val_score
import nashpy
import warnings
warnings.filterwarnings('ignore')

# NE
def add_NE(features):
    UNEs, NE_cnts = [], []
    for i in range(features.shape[0]):
        payoffs = np.array(features.iloc[i]) 
        R = payoffs[:9].reshape(3,3)
        C = payoffs[9:18].reshape(3,3)
        rps = nashpy.Game(R, C)
        eqs = list(rps.support_enumeration()) # could be unique or multiple (PNE MNE)
        UNE = list(np.concatenate(eqs[0])) if len(eqs)==1 else list(np.zeros(6))
        NE_cnt = len(eqs)    
        UNEs.append(UNE)
        NE_cnts.append(NE_cnt)

    # append to features
    names = ['UNE_r1', 'UNE_r2','UNE_r3','UNE_c1','UNE_c2','UNE_c3']

    for i in range(6):
        features[names[i]] = [UNE[i] for UNE in UNEs]
    features['NE_cnts'] = NE_cnts
    return None

# maxmax
def add_maxmax(features):
    features['max_max'] = features.iloc[:,:9].idxmax(axis=1).apply(lambda x: int(x[1]))
    features = pd.get_dummies(features, columns=['max_max'], drop_first=True)
    return None

# minimax
def grid_form(row):
    return np.array(row).reshape(3, 3)

def is_argmax(arr, i):
    return arr[i] == max(arr)
               
def is_min_max(row, i):
    grid = grid_form(row)
    mins = np.min(grid, axis=0)
    if is_argmax(mins, i - 1):
        return 1
    return 0

def add_minimax(features):
    for i in range(1,4):
        features['min_max_{}'.format(i)] = 0
        for j in range(len(features)):
            features.loc[j,'min_max_{}'.format(i)] = is_min_max(features.iloc[j, :9], i)
    return None

## add maximum payoff for both agents feature
def is_max_altruism(row, i):
    print(row)
    total_welfare = [row[i] + row[i + 9] for i in range(9)]
    if is_argmax(total_welfare, i):
        return 1
    return 0    

def add_maximin(features):
    for i in range(1,4):
        features['altruism_{}'.format(i)] = 0
        for j in range(len(features)):
            features.loc[j,'altruism_{}'.format(i)] = is_min_max(features.iloc[j, :9], i)
    return None

def _add_all_features(features):
    add_NE(features)
    add_maxmax(features)
    add_minimax(features)
    add_maximin(features)
    return None



if __name__ == "__main__":
    features = pd.read_csv('hb_train_feature.csv')
    truths = pd.read_csv('hb_train_truth.csv')

    _add_all_features(features)

    mlp = sk.neural_network.MLPRegressor(solver='lbfgs', alpha = 0.0005, random_state=1, max_iter=1000000000, verbose=True)
    mlp.fit(features, truths[['f1', 'f2', 'f3']])
    # create hb_test_pred
    test_features = pd.read_csv('hb_test_feature.csv')
    _add_all_features(test_features)
    scaler = sk.preprocessing.StandardScaler()
    scaler.fit(test_features)
    scaler.transform(test_features)
    output = pd.DataFrame(mlp.predict(test_features))
    output['action'] = output.idxmax(axis=1).apply(lambda x: x + 1)
    output.columns = ['f1', 'f2', 'f3', 'action']

    output.to_csv('hb_test_pred.csv', sep=',', encoding='utf-8', index=False)
    print(output)
    print("Pred written to hb_test_pred.csv")

