#!/usr/bin/env python 

import json 

if __name__ == '__main__':

    config = {} 
    
    config['augment_training_data'] = False 
    config['data_directory'] = '/Users/davidriser/Dropbox/coding-playground/python/airbnb/data'
    config['model_directory'] = '/Users/davidriser/Dropbox/coding-playground/python/airbnb/models'

    with open('config.json', 'w') as output:
        json.dump(config, output, indent = 4)
