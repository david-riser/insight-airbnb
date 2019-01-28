#!/usr/bin/env python 

import numpy as np
import pandas as pd

from sklearn.externals import joblib

if __name__ == '__main__':

    # Input data 
    data = pd.read_csv('./data/processed/redfin_boston.csv')
    model_name = 'random_forest'
    n_folds = 5

    models = []
    for index in range(n_folds):
        models.append(
            joblib.load('./models/{}_{}.joblib'.format(model_name, index))
        )

    # Kill the horrible name 
    kill_index = -1
    for col_index, col in enumerate(data.columns):
        if 'URL' in col:
            kill_index = col_index

    if kill_index > -1:
        new_cols = list(data.columns)
        new_cols[kill_index] = 'URL'
        data.columns = new_cols

    # Predict these houses
    features = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'crime_index']
    keep_cols = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'crime_index', 'PRICE']
    keep_cols.append('URL')

    for col in data.columns:
        if 'dist' in col:
            features.append(col)
            keep_cols.append(col)

    # Basic cleanup 
    data = data[keep_cols]
    data.dropna(how = 'any', inplace = True)

    data['airbnb_price'] = np.zeros(len(data))
    for model in models:
        data['airbnb_price'] += model.predict(data[features].values) / float(n_folds)


    # Save output 
    data.to_csv('./data/predictions/predictions.csv', index = False)

    print(data.sort_values('airbnb_price', ascending = False))
