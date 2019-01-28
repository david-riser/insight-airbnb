#!/usr/bin/env python 

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

from sklearn.ensemble import RandomForestRegressor 
from sklearn.externals import joblib
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error


def train_kfold(x, y, k = 5, model_name = 'unnamed', builder = RandomForestRegressor, params = None):
    ''' Train and CV a model built by builder. '''

    # Setup 5 fold CV
    kf = KFold(n_splits = k)

    fold_index = 0
    for train_index, valid_index in kf.split(x):

        print('Starting fold {}'.format(fold_index))

        # Setup simple model
        model = builder(**params)
        model.fit(x[train_index], y[train_index])

        y_pred = model.predict(x[valid_index])

        # Metric report
        print('Validation RMSE = {}'.format(np.sqrt(mean_squared_error(y_pred, y[valid_index]))))

        # Save Model
        output_model_name = './models/{}_{}.joblib'.format(model_name, fold_index)
        print('Saving model to {}.'.format(output_model_name))

        joblib.dump(model, output_model_name)

        fold_index += 1

if __name__ == '__main__':

    # Load clean datasets
    airbnb_data = pd.read_csv('./data/processed/airbnb.csv')
    redfin_data = pd.read_csv('./data/processed/redfin_boston.csv')

    # Get training and testing
    features = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'crime_index']

    # Add distance features
    for col in airbnb_data.columns:
        if 'dist' in col:
            features.append(col)

    x = airbnb_data[features].values
    y = airbnb_data['price'].values

    rf_params = {
        'n_jobs' : -1,
        'n_estimators' : 100
    }

    elastic_net_params = {
        'alpha' : 0.01
    }

    train_kfold(
        x = x,
        y = y,
        k = 5,
        model_name = 'random_forest',
        builder = RandomForestRegressor,
        params = rf_params
    )

    train_kfold(
        x = x,
        y = y,
        k = 5,
        model_name = 'elastic_net',
        builder = ElasticNet,
        params = elastic_net_params
    )