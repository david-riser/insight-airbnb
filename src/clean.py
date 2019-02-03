#!/usr/bin/env python 

import json 
import pandas as pd
import utils

from sklearn.neighbors import KernelDensity

def create_crime_kde(data, bandwidth = 0.001):

    ''' Load and clean crime dataset 
    and return a Scikit Learn KernelDensity 
    estimation object. '''

    # setup kde 
    kde = KernelDensity(
        kernel = 'gaussian',
        bandwidth = bandwidth
        )

    kde.fit(
        data[['latitude','longitude']].values
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

    # Load Config 
    with open('./config/config.json') as input_file:
        config = json.load(input_file)

        # Load Datasets
        airbnb_data = pd.read_csv('./data/raw/airbnb.csv')
        redfin_data = pd.read_csv('./data/raw/redfin_boston.csv')
        crimes_data = pd.read_csv('./data/raw/crime.csv', encoding = 'latin-1')
        
        # Clean data
        airbnb_data = utils.clean_airbnb_dataset(airbnb_data, config)
        redfin_data = utils.clean_redfin_dataset(redfin_data, config)
        crimes_data = utils.clean_crimes_dataset(crimes_data, config)
        
        # Build crime KDE
        kde = create_crime_kde(crimes_data, 0.1)
        
        # Add crime scores to data
        utils.add_crime_index(airbnb_data, kde)
        utils.add_crime_index(redfin_data, kde)
        
        # Get Attractions
        attractions = build_list_of_attractions()
        utils.add_attraction_distances(airbnb_data, attractions)
        utils.add_attraction_distances(redfin_data, attractions)
        
        # Save cleaned data
        airbnb_data.to_csv('./data/processed/airbnb.csv', index = False)
        redfin_data.to_csv('./data/processed/redfin_boston.csv', index = False)
        
