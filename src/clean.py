#!/usr/bin/env python 

import json 
import pandas as pd
import utils

from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

def create_clusters(airbnb_data, redfin_data, k = 8):
    ''' Cluster latitude and longitude into k clusters
    using the kmeans alogirthm implemented by sklearn.

    Arguments: 
    ----------

    airbnb_data: dataframe containing the columns latitude and 
    longitude for airbnb properties 

    redfin_data: optional dataframe containing columns latitude
    and longitude for redfin properties

    k: the number of clusters to create 

    Returns: 
    --------
    Nothing, the dataframes are altered. 
    '''

    kmeans = KMeans(k)
    airbnb_data['cluster_index'] = kmeans.fit_predict(airbnb_data[['latitude', 'longitude']].values)

    if redfin_data is not None:
        redfin_data['cluster_index'] = kmeans.predict(redfin_data[['latitude', 'longitude']].values)

def create_crime_kde(data, bandwidth = 0.002, kernel_type = 'exponential'):
    ''' Create a kernel density estimate of the occurances in the 
    dataframe (named data) as a function of latitude and longitude 
    variables.  

    Arguments: 
    ----------
    data: the dataframe which contains columns latitude and longitude 
    bandwidth: kernel bandwidth parameter for the kernel density estimate 
    kernel_type: the kernel type (gaussian, exponential, ...) used in the 
    density estimate 
    
    Returns: 
    --------
    kde: a sklearn.neighbors.KernelDenstiy object that can be used to query
    the density for a (latitude, longitude) pair. 

    '''

    if not isinstance(data, pd.DataFrame):
        raise TypeError('Data must be a pandas dataframe')
    elif 'latitude' not in data.columns or 'longitude' not in data.columns:
        raise KeyError('Dataframe must contain columns latitude and longitude')

    # setup kde 
    kde = KernelDensity(
        kernel = kernel_type,
        bandwidth = bandwidth
        )

    kde.fit(
        data[['latitude','longitude']].values
        )

    return kde 

def build_list_of_attractions():
    ''' Return a python list of Boston attractions addresses. 

    Arguements: 
    None 

    Returns: 
    attractions: A python list of addresses of some things that I think 
    might be important attractions. 

    '''

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

def load_t_stations_to_kdtree(file_path = './data/processed/mbta_stations.csv'):
    ''' Load and fill a KDTree with the MBTA stations. 

    Arguments: 
    ---------
    file_path: path to the MBTA data produced by process_t_stations.py
    

    Returns: 
    --------
    kdtree: A sklearn.neighbors.KDTree object that can be used to query 
    for the closest mbta stations. 

    '''
    data = pd.read_csv(file_path)
    kdtree = KDTree(data[['latitude', 'longitude']].values)
    return kdtree

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
        kde = create_crime_kde(crimes_data, 0.002)

        # Build KDTree that contains subway stations 
        kdtree = load_t_stations_to_kdtree()

        # Add crime scores to data
        utils.add_crime_index(airbnb_data, kde)
        utils.add_crime_index(redfin_data, kde)

        # Add Clusters 
        create_clusters(airbnb_data, redfin_data, 4)
        
        # Get Attractions
        attractions = build_list_of_attractions()
        utils.add_attraction_distances(airbnb_data, attractions)
        utils.add_attraction_distances(redfin_data, attractions)

        # Add closest MBTA stop 
        utils.add_closest_t_stop(airbnb_data, kdtree)
        utils.add_closest_t_stop(redfin_data, kdtree)

        # Save cleaned data
        airbnb_data.to_csv('./data/processed/airbnb.csv', index = False)
        redfin_data.to_csv('./data/processed/redfin_boston.csv', index = False)
        
