#!/usr/bin/env python 

import json

if __name__ == '__main__':

    config = {
        'max_nightly_price' : 800, 
        'max_bedrooms' : 6,
        'min_latitude' : 35,
        'max_latitude' : 45,
        'min_longitude' : -72,
        'max_longitude' : -70
        }
    
    with open('./config/config.json', 'w') as output_file:
        json.dump(config, output_file, indent = 4)
