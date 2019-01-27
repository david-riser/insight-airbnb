#!/usr/bin/env python 

import numpy as np 
import pandas as pd

def clean_dataset(data):
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
    data = data[data['room_type'] == 'Entire home/apt']
    data = data[data['price'] < 800]
    data = data[data['bedrooms'] < 8]

    # Remove nan 
    data.dropna(how = 'any', inplace = True)

    return data 


