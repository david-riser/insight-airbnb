#!/usr/bin/env python 

import pandas as pd 
import yaml 

def create_data_structure():
    data_dict = {}
    data_dict['station'] = []
    data_dict['latitude'] = []
    data_dict['longitude'] = []
    return data_dict

def fill_data_structure(data_dict, station):
    data_dict['station'].append(station['title'])
    data_dict['latitude'].append(station['latitude'])
    data_dict['longitude'].append(station['longitude'])

if __name__ == '__main__':

    input_file = './data/raw/mbta.yaml'

    data_dict = create_data_structure()

    with open(input_file, 'r') as stream:
        yaml_data = yaml.load(stream)

        for list_item in yaml_data:
            for station in list_item['stations']:
                if 'title' in station.keys():
                    fill_data_structure(data_dict, station)
    
    # Spit out the info 
    pd.DataFrame(data_dict).to_csv(
        './data/processed/mbta_stations.csv', 
        index = False
        )
