#!/usr/bin/env python 

import argparse
import json
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import utils

from geopy.geocoders import Nominatim 
from geopy import distance 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KernelDensity

def load_dataset(data_path, sample_size):
    return pd.read_csv(data_path, nrows = sample_size) 

def create_crime_kde(config):

    ''' Load and clean crime dataset 
    and return a Scikit Learn KernelDensity 
    estimation object. '''

    data = pd.read_csv(
        config['data_directory'] + '/crime.csv',
        encoding = 'latin-1'
        )

    print('Loaded crime data with shape {}.'.format(
            data.shape
            ))

    # setup kde 
    kde = KernelDensity(
        kernel = 'gaussian',
        bandwidth = 0.01
        )

    crime_codes = [3115, 1402, 3831, 802, 3301, 2647]
    data['is_considered_in_model'] = data['OFFENSE_CODE'].apply(lambda x: x in crime_codes)
    data_subset = data[data['is_considered_in_model'] == True]
    data_subset = data_subset[data_subset['Lat'] > 39]
    data_subset = data_subset[data_subset['Long'] < 20]

    kde.fit(
        data_subset[['Lat','Long']].values
        )

    return kde 

def build_list_of_attractions():
    attractions = [
        '145 Dartmouth St, Boston, MA 02116',
        '1400 Massachusetts Ave, Cambridge, MA 02138',
        '1 Science Park, Boston, MA 02114',
        '4 S Market St, Boston, MA 02109',
        '560 Boylston, Boston, MA 02116',
        '465 Huntington Ave, Boston, MA 02115',
        '19 N Square, Boston, MA 02113',
        '30 Germania St, Boston, MA 02130',
        '210 Union St, Braintree, MA 02184'
        ]
    return attractions 

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required = True)
    args = ap.parse_args()

    with open(args.config, 'r') as inputfile:
        config = json.load(inputfile)

        # Dataset setup 
        data_path = '/Users/davidriser/Data/air_bnb_boston/listings.csv'
        sample_size = None
        
        # Load data 
        df = load_dataset(data_path, sample_size)
        
        # Clean data 
        df = utils.clean_dataset(df)
    
        # ---------------------------
        # Build Crime KDE 
        # ---------------------------
        kde = create_crime_kde(config)
        df['crime_index'] = kde.score_samples(
            df[['latitude', 'longitude']].values
            )
        
        # Get Attractions 
        attractions = build_list_of_attractions() 
        
        attraction_locs = []
        for attr in attractions:
            locator = Nominatim()
            location = locator.geocode(attr)

            if location is not None:
                attraction_locs.append(np.array([location.latitude, location.longitude]))        

        # Calculate distance to the attractions 
        for index, attr in enumerate(attraction_locs):
            df['dist_{}'.format(index)] = np.zeros(len(df))

            for row_index, row in df.iterrows():
                df['dist_{}'.format(index)][row_index] = distance.distance(attr, (row['latitude'], row['longitude'])).miles
            
        print(df.head())

        # Get training and testing 
        features = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'crime_index']

        for col in df.columns:
            if 'dist' in col:
                features.append(col)
        
        x_train, x_test, y_train, y_test = train_test_split(
            df[features].values, df['price'].values
            )

        # Setup simple model 
        rf = RandomForestRegressor(n_jobs = -1, n_estimators = 50)
        rf.fit(x_train, y_train)

        # plot 
        plt.barh(
            features, 
            rf.feature_importances_
            )
        plt.savefig('image/feature_importance.png', bbox_inches = 'tight')

        y_pred = rf.predict(x_test)

        # Metric report 
        print('Validation RMSE = {}'.format(
                np.sqrt(mean_squared_error(y_pred, y_test))
                ))

        # Save Model 
        output_model_name = config['model_directory'] + '/random_forest.joblib'
        print('Saving model to {}.'.format(
                output_model_name
                ))

        joblib.dump(rf, output_model_name)


