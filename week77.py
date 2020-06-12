import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import time
import datetime

def gradboost():
    data = pd.read_csv('features.csv', index_col='match_id')
    v = data['radiant_win']
    y = v.to_numpy()
    data_1 = data.drop(['duration',
                        'radiant_win',
                        'tower_status_radiant',
                        'tower_status_dire',
                        'barracks_status_radiant',
                        'barracks_status_dire'], axis=1)
    print('поиск данных с пропусками: ')
    for i in range(1, 12):
        print(data_1.count()[(10*(i-1)):10*i])
    data_2 = data_1.fillna(0)
    print(data_2.count()[-20:])
    ab = data_2
    x = ab.to_numpy()

    cval = KFold(n_splits=5, shuffle=True, random_state=1)
    for p in [30]:
        su = 0
        start_time = datetime.datetime.now()
        for train, test in cval.split(x, y):
            x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
            model = GradientBoostingClassifier(n_estimators=p, random_state=1)
            model.fit(x[train], y[train])
            pred = model.predict_proba(x[test])[:, 1]
            sc = roc_auc_score(y[test], pred)
            su = su + sc
        print('качество работы алгоритма ', su / 5.0)
        print('Time elapsed for tree:', p, ' ', datetime.datetime.now() - start_time)
        print('\n')

def log_reg():
    scal = StandardScaler()
    data = pd.read_csv('features.csv', index_col='match_id')
    print(data.shape)
    v = data['radiant_win']
    y = v.to_numpy()

    print(y.shape)
    data_1 = data.drop(['duration',
                        'radiant_win',
                        'tower_status_radiant',
                        'tower_status_dire',
                        'barracks_status_radiant',
                        'barracks_status_dire'], axis=1)
    data_2 = data_1.fillna(0)
    ab = data_2
    xx = ab.to_numpy()
    x = scal.fit_transform(xx)
    cval = KFold(n_splits=5, shuffle=True, random_state=1)


    c_values = np.linspace(0.0001, 2, 20)
    for i in c_values:
        su = 0
        start_time = datetime.datetime.now()
        for train, test in cval.split(x, y):
            x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
            model = LogisticRegression(penalty='l2', C=i)
            model.fit(x[train], y[train])
            pred = model.predict_proba(x[test])[:, 1]
            sc = roc_auc_score(y[test], pred)
            su = su + sc

        print('качество работы алгоритма ', su / 5.0)
        print('Time elapsed for :', i, ' ', datetime.datetime.now() - start_time)
        print('\n')

def log_reg_bez_cat():
    scal = StandardScaler()
    data = pd.read_csv('features.csv', index_col='match_id')
    v = data['radiant_win']
    y = v.to_numpy()
    data_1 = data.drop(['duration',
                        'radiant_win',
                        'tower_status_radiant',
                        'tower_status_dire',
                        'barracks_status_radiant',
                        'barracks_status_dire',
                        'lobby_type',
                        'r1_hero',
                        'r2_hero',
                        'r3_hero',
                        'r4_hero',
                        'r5_hero',
                        'd1_hero',
                        'd2_hero',
                        'd3_hero',
                        'd4_hero',
                        'd5_hero'], axis=1)
    print('schitaem unikalnyx geroev:')
    heroes = list()
    for i in ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero','d5_hero']:
        heroes.append(len(data[i].unique()))
    print('unikalnyx geroev:', max(heroes))

    print(data_1.shape, 'shape bez kategor i udal')
    data_2 = data_1.fillna(0)
    ab = data_2
    xx = ab.to_numpy()
    x = scal.fit_transform(xx)
    cval = KFold(n_splits=5, shuffle=True, random_state=1)
    start_time = datetime.datetime.now()

    c_values = np.linspace(0.0001, 2, 20)
    for i in c_values:
        su = 0
        for train, test in cval.split(x, y):
            x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
            model = LogisticRegression(penalty='l2', C=i)
            model.fit(x[train], y[train])
            pred = model.predict_proba(x[test])[:, 1]
            sc = roc_auc_score(y[test], pred)
            su = su + sc

        print('качество работы алгоритма ', su / 5.0)
        print('Time elapsed for :', i, ' ', datetime.datetime.now() - start_time)
        print('\n')

def meshok():
    scal = StandardScaler()
    data = pd.read_csv('features.csv', index_col='match_id')

    v = data['radiant_win']
    y = v.to_numpy()
    data_mesh = data.loc[:, ['lobby_type',
                             'r1_hero',
                             'r2_hero',
                             'r3_hero',
                             'r4_hero',
                             'r5_hero',
                             'd1_hero',
                             'd2_hero',
                             'd3_hero',
                             'd4_hero',
                             'd5_hero']]
    print(data_mesh.shape)
    data_1 = data.drop(['duration',
                        'radiant_win',
                        'tower_status_radiant',
                        'tower_status_dire',
                        'barracks_status_radiant',
                        'barracks_status_dire',
                        'lobby_type',
                        'r1_hero',
                        'r2_hero',
                        'r3_hero',
                        'r4_hero',
                        'r5_hero',
                        'd1_hero',
                        'd2_hero',
                        'd3_hero',
                        'd4_hero',
                        'd5_hero'], axis=1)
    data_2 = data_1.fillna(0)
    ab = data_2
    xx = ab.to_numpy()


    N = int(112)

    X_pick = np.zeros((data_mesh.shape[0], N))
    for i, match_id in enumerate(data_mesh.index):
        for p in range(5):
            X_pick[i, data_mesh.loc[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            X_pick[i, data_mesh.loc[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    xxx = np.hstack((xx, X_pick))
    x = scal.fit_transform(xxx)
    #print(x)
    cval = KFold(n_splits=5, shuffle=True, random_state=1)


    c_values = np.linspace(0.0001, 2, 20)
    for i in c_values:
        su = 0
        start_time = datetime.datetime.now()
        for train, test in cval.split(x, y):
            x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]
            model = LogisticRegression(penalty='l2', C=i)
            model.fit(x[train], y[train])
            pred = model.predict_proba(x_test)[:, 1]
            sc = roc_auc_score(y[test], pred)
            su = su + sc

        print('качество работы алгоритма ', su / 5.0)
        print('Time elapsed for :', i, ' ', datetime.datetime.now() - start_time)
        print('\n')

def test_vib():
    scal = StandardScaler()
    data = pd.read_csv('features.csv', index_col='match_id')

    test_data = pd.read_csv('features_test.csv', index_col='match_id')
    data_mesh_test = data.loc[:, ['lobby_type',
                                  'r1_hero',
                                  'r2_hero',
                                  'r3_hero',
                                  'r4_hero',
                                  'r5_hero',
                                  'd1_hero',
                                  'd2_hero',
                                  'd3_hero',
                                  'd4_hero',
                                  'd5_hero']]
    data_1_test = data.drop(['duration',
                             'radiant_win',
                             'tower_status_radiant',
                             'tower_status_dire',
                             'barracks_status_radiant',
                             'barracks_status_dire',
                             'lobby_type',
                             'r1_hero',
                             'r2_hero',
                             'r3_hero',
                             'r4_hero',
                             'r5_hero',
                             'd1_hero',
                             'd2_hero',
                             'd3_hero',
                             'd4_hero',
                             'd5_hero'], axis=1)
    data_2_test = data_1_test.fillna(0)
    abcc = data_2_test
    qqq = abcc.to_numpy()

    N = int(112)
    X_pick_test = np.zeros((data_mesh_test.shape[0], N))
    for i, match_id in enumerate(data_mesh_test.index):
        for p in range(5):
            X_pick_test[i, data_mesh_test.loc[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            X_pick_test[i, data_mesh_test.loc[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    xxxx = np.hstack((qqq, X_pick_test))
    x_test = scal.fit_transform(xxxx)

    v = data['radiant_win']
    y = v.to_numpy()
    data_mesh = data.loc[:, ['lobby_type',
                             'r1_hero',
                             'r2_hero',
                             'r3_hero',
                             'r4_hero',
                             'r5_hero',
                             'd1_hero',
                             'd2_hero',
                             'd3_hero',
                             'd4_hero',
                             'd5_hero']]

    data_1 = data.drop(['duration',
                        'radiant_win',
                        'tower_status_radiant',
                        'tower_status_dire',
                        'barracks_status_radiant',
                        'barracks_status_dire',
                        'lobby_type',
                        'r1_hero',
                        'r2_hero',
                        'r3_hero',
                        'r4_hero',
                        'r5_hero',
                        'd1_hero',
                        'd2_hero',
                        'd3_hero',
                        'd4_hero',
                        'd5_hero'], axis=1)
    data_2 = data_1.fillna(0)
    ab = data_2
    xx = ab.to_numpy()



    X_pick = np.zeros((data_mesh.shape[0], N))
    for i, match_id in enumerate(data_mesh.index):
        for p in range(5):
            X_pick[i, data_mesh.loc[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            X_pick[i, data_mesh.loc[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    xxx = np.hstack((xx, X_pick))
    x = scal.fit_transform(xxx)





    model = LogisticRegression(penalty='l2', C=0.1)
    model.fit(x, y)
    pred = model.predict_proba(x_test)[:, 1]
    print('минимальная вероятность победы Radiant', pred.min())
    print('максимальная вероятность победы Radiant', pred.max())
    print(pred)



test_vib()