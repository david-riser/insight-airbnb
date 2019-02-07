#!/usr/bin/env python 

import numpy as np
import pandas as pd

from sklearn.externals import joblib

if __name__ == '__main__':

    # Input data 
    data = pd.read_csv('./data/processed/redfin_boston.csv')
    model_name = 'random_forest'
    model = joblib.load('./models/{}_log_trans.pkl'.format(model_name))
    

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
    features = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'crime_index', 
                'cluster_index', 'mbta_1', 'mbta_2', 'mbta_3']
    keep_cols = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'crime_index', 'price', 'cluster_index', 'mbta_1', 'mbta_2', 'mbta_3']
    keep_cols.append('URL')

    for col in data.columns:
        if 'dist' in col:
            features.append(col)
            keep_cols.append(col)

    # Basic cleanup 
    data = data[keep_cols]
    data.dropna(how = 'any', inplace = True)

    # Standardization 
    ss = joblib.load('./models/standard_scaler.pkl')
    xp = ss.transform(data[features].values)

    data['airbnb_log_price'] = model.predict(xp)

    # I have to kill anything that comes out zero or negative
    # due to the nature of the transformation. 
    data = data[data['airbnb_log_price'] > 0]

    data['airbnb_price'] = np.exp(data['airbnb_log_price'])

    # Save output 
    data.to_csv('./data/predictions/predictions.csv', index = False)
 
    print(data.sort_values('airbnb_price', ascending = False))
