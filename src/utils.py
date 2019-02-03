#!/usr/bin/env python 

import numpy as np 
import pandas as pd
import tqdm
import warnings
warnings.filterwarnings('ignore')


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
    data = data[data['room_type'] == 'Entire home/apt']
    data = data[data['price'] < float(config['max_nightly_price'])]
    data = data[data['bedrooms'] < int(config['max_bedrooms'])]

    # Remove nan 
    data.dropna(how = 'any', inplace = True)

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
    important_crimes = ['Investigate Person', 'Drug Violation', 'Simple Assault', 
                        'Aggravated Assault']

    data['is_important_crime'] = data['OFFENSE_CODE_GROUP'].apply(lambda x: x in important_crimes)
    data = data[data['is_important_crime'] > 0]
    data = data[data['latitude'] > float(config['min_latitude'])]
    data = data[data['longitude'] < float(config['max_longitude'])]
    data = data[['OFFENSE_CODE', 'OFFENSE_CODE_GROUP', 'latitude', 'longitude']]
    data.dropna(how = 'any', inplace = True)

    return data

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
    data = data[data['bedrooms'] < int(config['max_bedrooms'])]
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


def add_closest_t_stop(data, kdtree):
    ''' Add metro stops distance from kdtree '''

    data['dist_mbta_1'] = np.zeros(len(data))
    data['dist_mbta_2'] = np.zeros(len(data))
    data['dist_mbta_3'] = np.zeros(len(data))

    for row_index, row in tqdm.tqdm(data.iterrows(), total = len(data)):
        dist, ind = kdtree.query(
            np.array([row['latitude'], row['longitude']]).reshape(1, -1), k = 3
            )

        data['dist_mbta_1'][row_index] = dist[0][0]
        data['dist_mbta_2'][row_index] = dist[0][1]
        data['dist_mbta_3'][row_index] = dist[0][2]
    
