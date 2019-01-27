#!/usr/bin/env python 

''' 

File:   kde.py 
Author: David M. Riser 
Date:   January 27, 2019 


This file is used to create a 
Kernel density estimate of the 
crime in Boston. 
'''

import argparse
import json
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import utils
import os 

from geopy.geocoders import Nominatim 
from geopy import distance 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KernelDensity

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

if __name__ == '__main__':

    base_directory = os.path.abspath('./')

    print('Base directory {}'.format(
            base_directory
            ))

#    kde = create_crime_kde(config)
#    df['crime_index'] = kde.score_samples(
#        df[['latitude', 'longitude']].values
#        )
    



