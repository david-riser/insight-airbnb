#!/usr/bin/env python 

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

from sklearn.ensemble import RandomForestRegressor 
from sklearn.externals import joblib
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def train_kfold(x, y, k = 5, model_name = 'unnamed', builder = RandomForestRegressor, params = None):
    ''' Train and CV a model built by builder. '''

    # Setup 5 fold CV
    kf = KFold(n_splits = k)

    # Out of fold predictions to 
    # study performance
    oof_predictions = np.zeros(len(y))

    fold_index = 0
    for train_index, valid_index in kf.split(x):

        print('[{}] Starting fold {}'.format(model_name, fold_index))

        # Setup simple model
        model = builder(**params)
        model.fit(x[train_index], y[train_index])

        y_pred = model.predict(x[valid_index])
        oof_predictions[valid_index] = y_pred 

        # Metric report
        print('[{}] Training RMSE = {}'.format(model_name, np.sqrt(mean_squared_error(model.predict(x[train_index]), y[train_index]))))
        print('[{}] Validation RMSE = {}'.format(model_name, np.sqrt(mean_squared_error(y_pred, y[valid_index]))))

        # Save Model
        output_model_name = './models/{}_{}.joblib'.format(model_name, fold_index)
        print('[{}] Saving model to {}.'.format(model_name, output_model_name))

        joblib.dump(model, output_model_name)

        fold_index += 1

        if model_name == 'elastic_net':
            print('[{}] '.format(model_name), model.coef_)

    return oof_predictions 

if __name__ == '__main__':

    # Load clean datasets
    airbnb_data = pd.read_csv('./data/processed/airbnb.csv')
    redfin_data = pd.read_csv('./data/processed/redfin_boston.csv')

    # Get training and testing
    features = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'crime_index', 'cluster_index']

    # Add distance features
    for col in airbnb_data.columns:
        if 'dist' in col:
            features.append(col)

    x = airbnb_data[features].values
    y = airbnb_data['price'].values

    # Scale 
    ss = StandardScaler() 
    x = ss.fit_transform(x)

    rf_params = {
        'n_jobs' : -1,
        'n_estimators' : 10
    }

    elastic_net_params = {
        'fit_intercept' : True,
        'alpha' : 1.0
    }

    svr_params = {
        'kernel' : 'rbf',
        'gamma' : 'scale',
        'degree' : 3,
        'C' : 1.0,
        'epsilon' : 0.1
    }

    knn_params = {
        'n_neighbors' : 5
        }

    oof_preds = train_kfold(
        x = x,
        y = y,
        k = 5,
        model_name = 'random_forest',
        builder = RandomForestRegressor,
        params = rf_params
    )

    # Save this to analyze later 
    airbnb_data['oof_pred'] = oof_preds
    airbnb_data.to_csv('./data/predictions/oof_random_forest.csv', index = False)

    oof_preds = train_kfold(
        x = x,
        y = y,
        k = 5,
        model_name = 'svr',
        builder = SVR,
        params = svr_params
    )

    # Save this to analyze later 
    airbnb_data['oof_pred'] = oof_preds
    airbnb_data.to_csv('./data/predictions/oof_svr.csv', index = False)

    oof_preds = train_kfold(
        x = x,
        y = y,
        k = 5,
        model_name = 'knn',
        builder = KNeighborsRegressor,
        params = knn_params
    )

    # Save this to analyze later 
    airbnb_data['oof_pred'] = oof_preds
    airbnb_data.to_csv('./data/predictions/oof_knn.csv', index = False)

    # For a linear model let's remove lat/long 
    features.remove('latitude')
    features.remove('longitude')
    x = airbnb_data[features].values

    print('Moving to elastic_net w/ features: ', features)
    oof_preds = train_kfold(
        x = x,
        y = y,
        k = 5,
        model_name = 'elastic_net',
        builder = ElasticNet,
        params = elastic_net_params
    )

    # Save this to analyze later 
    airbnb_data['oof_pred'] = oof_preds
    airbnb_data.to_csv('./data/predictions/oof_elastic_net.csv', index = False)

    
