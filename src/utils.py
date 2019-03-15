#!/usr/bin/env python 

import numpy as np 
import pandas as pd
import tqdm
import warnings
warnings.filterwarnings('ignore')

from geopy.geocoders import Nominatim
from geopy import distance

def clean_airbnb_dataset(data, config):
    ''' Apply all cleaning operations to the dataset
    from airbnb.  These operations were discovered in 
    the data EDA notebooks. 

    Arguments: 
    ----------
    data: A pandas.DataFrame that contains the full, uncleaned airbnb dataset
    config: A dictionary that defines numerical bounds for filtering variables

    Returns: 
    -------- 
    data: The airbnb dataset cleaned, still a pandas.DataFrame. 

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
    data = data[data['price'] > float(config['min_nightly_price'])]
    data = data[data['price'] < float(config['max_nightly_price'])]
    data = data[data['bedrooms'] < int(config['max_bedrooms'])]

    # Remove nan 
    data.dropna(how = 'any', inplace = True)

    return data 


def clean_crimes_dataset(data, config):
    ''' Apply all cleaning operations to the dataset
    from airbnb.  These operations were discovered in 
    the data EDA notebooks. 

    Arguments: 
    ----------
    data: A pandas.DataFrame that contains the full, uncleaned crimes dataset
    config: A dictionary that defines numerical bounds for filtering variables

    Returns: 
    -------- 
    data: The crimes dataset cleaned, still a pandas.DataFrame. 

    '''
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
    ''' Apply all cleaning operations to the dataset
    from redfin.  These operations were discovered in 
    the data EDA notebooks. 

    Arguments: 
    ----------
    data: A pandas.DataFrame that contains the full, uncleaned redfin dataset
    config: A dictionary that defines numerical bounds for filtering variables

    Returns: 
    -------- 
    data: The redfin dataset cleaned, still a pandas.DataFrame. 

    '''

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
    ''' Add a column to the dataframe data that describes the 
    crime index based on the kernel density estimate. 

    Arguements: 
    data: A pandas.DataFrame that contains the colunms 
    latitude and longitude. 
    kde: A sklearn.neighbors.KernelDensity object that already 
    has the points loaded.  

    Returns: 
    None, the column is appened in the function. 

    '''

    data['crime_index'] = kde.score_samples(
        data[['latitude', 'longitude']].values
    )

def add_attraction_distances(data, attractions):
    ''' Find the distance on the surface of the earth (not by traversing
    roads) between each house and each attraction.  This is done using the 
    geopy module. 

    Arguments: 
    ----------
    data: A pandas.DataFrame that contains the columns 
    latitude and longitude. 
    attractions: A python list of addresses that will be 
    used to compute distance.  For each address, a distance 
    variable named dist_i will be added to the dataframe, here i 
    refers to the i-th attraction in the list. 


    Returns: 
    None, the columns are added to the dataframe. 
    ''' 

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
    ''' Add top 3 metro stops distance and code from kdtree.

    Arguments: 
    ----------
    data: A pandas.DataFrame which contains columns latitude and longitude for
    a list of houses. 

    kdtree: A sklearn.neighbors.KDTree object loaded with mbta stations 
    that will be queried. 
    
    Returns: 
    --------
    None, the columns are added to the dataframe. 

    '''

    data['dist_mbta_1'] = np.zeros(len(data))
    data['dist_mbta_2'] = np.zeros(len(data))
    data['dist_mbta_3'] = np.zeros(len(data))

    data['mbta_1'] = np.zeros(len(data))
    data['mbta_2'] = np.zeros(len(data))
    data['mbta_3'] = np.zeros(len(data))

    for row_index, row in tqdm.tqdm(data.iterrows(), total = len(data)):
        dist, ind = kdtree.query(
            np.array([row['latitude'], row['longitude']]).reshape(1, -1), k = 3
            )

        data['dist_mbta_1'][row_index] = dist[0][0]
        data['dist_mbta_2'][row_index] = dist[0][1]
        data['dist_mbta_3'][row_index] = dist[0][2]
        data['mbta_1'][row_index] = ind[0][0]
        data['mbta_2'][row_index] = ind[0][1]
        data['mbta_3'][row_index] = ind[0][2]
        
