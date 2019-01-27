#!/usr/bin/env python 

import numpy as np
import pandas as pd 

from sklearn.externals import joblib

if __name__ == '__main__':

    # Input data 
    data_path = '../data/redfin_boston.csv'
    model_path = '../models/price_model_rf.joblib'
    sample_size = 100

    # Load data 
    data = pd.read_csv(data_path, nrows = sample_size)
    print(data.head())

    # Kill the horrible name 
    kill_index = -1
    for col_index, col in enumerate(data.columns):
        if 'URL' in col:
            kill_index = col_index

    if kill_index > -1:
        new_cols = list(data.columns)
        new_cols[kill_index] = 'URL'
        data.columns = new_cols

    
    # Load model 
    model = joblib.load(model_path)
    
    # Predict these houses
    features = ['BEDS', 'BATHS', 'LATITUDE', 'LONGITUDE']
    keep_cols = ['BEDS', 'BATHS', 'LATITUDE', 'LONGITUDE']
    #    keep_cols.append('URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)')
    keep_cols.append('URL')

    # Basic cleanup 
    data = data[keep_cols]
    data.dropna(how = 'any', inplace = True)
    print(data.head())

    data['airbnb_price'] = model.predict(data[features].values)
    print(data.head())

    # Save output 
    data.to_csv('../data/predictions.csv', index = False)

    print(data.sort_values('airbnb_price', ascending = False))
