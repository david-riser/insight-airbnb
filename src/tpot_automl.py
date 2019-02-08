#!/usr/bin/env python 

import pandas as pd
import tpot as tp

from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    data = pd.read_csv('./data/processed/airbnb.csv')
    
    # Define features 
    features = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'crime_index', 'cluster_index',
                'mbta_1', 'mbta_2', 'mbta_3']

    # Add distance features
    for col in data.columns:
        if 'dist' in col:
            features.append(col)

    x_train, x_test, y_train, y_test = train_test_split(
        data[features].values,
        data['price'].values
        )

    regressor = tp.TPOTRegressor(
        generations = 4,
        population_size = 40,
        verbosity = 2
        )

    
