#!/usr/bin/env python 

import numpy as np 
import pandas as pd
import tqdm

from geopy.geocoders import Nominatim
from geopy import distance

def clean_airbnb_dataset(data, config):
    '''

    This function accepts the airbnb dataframe
    and returns a cleaned copy that is ready to
    be split for training and testing. 

    '''

    # Drop cols we don't need or want. 
    keep_cols = [
        'latitude', 'longitude', 
        'beds', 'bedrooms', 'bathrooms', 'accomodates', 
        'price', 'room_type'
        ]

    for col in data.columns:
        if col not in keep_cols:
            data.drop(col, inplace = True, axis = 1)
        
    # Format price as a float 
    data['price'] = data['price'].replace( '[\$,)]','', regex=True).astype(float)

    # Apply some filtering options 
    print('Cleaning AirBnB data...')
    print(len(data))
    data = data[data['room_type'] == 'Entire home/apt']
    print(len(data))
    data = data[data['price'] < float(config['max_nightly_price'])]
    print(len(data))
    data = data[data['bedrooms'] < int(config['max_bedrooms'])]
    print(len(data))

    # Remove nan 
    data.dropna(how = 'any', inplace = True)
    print(len(data))

    return data 


def clean_crimes_dataset(data, config):

    # Standardize names of columns
    cols = list(data.columns)
    new_cols = []

    for col in cols:
        if col == 'Lat':
            new_cols.append('latitude')
        elif col == 'Long':
            new_cols.append('longitude')
        else:
            new_cols.append(col)

    data.columns = new_cols

    # Do cleaning
    crime_codes = [3115, 1402, 3831, 802, 3301, 2647]
    data['is_considered_in_model'] = data['OFFENSE_CODE'].apply(lambda x: x in crime_codes)
    data_subset = data[data['is_considered_in_model'] == True]
    print(len(data_subset))
    data_subset = data_subset[data_subset['latitude'] > float(config['min_latitude'])]
    print(len(data_subset))
    data_subset = data_subset[data_subset['longitude'] < float(config['max_longitude'])]
    print(len(data_subset))

    return data_subset

def clean_redfin_dataset(data, config):

    # Standardize names of columns
    cols = list(data.columns)
    new_cols = []

    for col in cols:
        if col == 'LATITUDE':
            new_cols.append('latitude')
        elif col == 'LONGITUDE':
            new_cols.append('longitude')
        elif col == 'BEDS':
            new_cols.append('bedrooms')
        elif col == 'BATHS':
            new_cols.append('bathrooms')
        elif col == 'PRICE':
            new_cols.append('price')
        else:
            new_cols.append(col)

    data.columns = new_cols
    
    print('Cleaning redfin data...')
    print(len(data))
    data = data[data['bedrooms'] < int(config['max_bedrooms'])]
    print(len(data))
    return data

def add_crime_index(data, kde):
    data['crime_index'] = kde.score_samples(
        data[['latitude', 'longitude']].values
    )

def add_attraction_distances(data, attractions):

    attraction_locs = []
    for attr in attractions:
        locator = Nominatim(user_agent = 'attraction_distance_agent')
        location = locator.geocode(attr)

        if location is not None:
            attraction_locs.append(np.array([location.latitude, location.longitude]))

    # Calculate distance to the attractions
    for index, attr in enumerate(attraction_locs):
        data['dist_{}'.format(index)] = np.zeros(len(data))

        for row_index, row in tqdm.tqdm(data.iterrows(), total = len(data)):
            data['dist_{}'.format(index)][row_index] = distance.distance(attr, (row['latitude'], row['longitude'])).miles



