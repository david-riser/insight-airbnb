#!/usr/bin/env python 

import pandas as pd 
import yaml 

if __name__ == '__main__':

    input_file = './data/raw/mbta.yaml'

    data_dict = {}
    data_dict['station'] = []
    data_dict['latitude'] = []
    data_dict['longitude'] = []

    with open(input_file, 'r') as stream:
        yaml_data = yaml.load(stream)

        for list_item in yaml_data:
            print('-' * 40)
            print(list_item['title'])

            for station in list_item['stations']:
                if 'title' in station.keys():
                    data_dict['station'].append(station['title'])
                    data_dict['latitude'].append(station['latitude'])
                    data_dict['longitude'].append(station['longitude'])


    
    # Spit out the info 
    data = pd.DataFrame(data_dict)
    data.to_csv('./data/processed/mbta_stations.csv', index = False)
