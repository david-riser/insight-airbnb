#!/usr/bin/env python 

import numpy as np 
import pandas as pd 
import pymc3 as pm 
import pickle 

if __name__ == '__main__':

    # Load predictions dataset 
    redfin_data = pd.read_csv('./data/processed/redfin_boston.csv')

    # Load pymc model 
    with open('./models/bayesian_model.pickle', 'rb') as buffer:
        input_data_dict = pickle.load(buffer)

        # Retrieve from dict the important pieces 
        trace, model = input_data_dict['trace'], input_data_dict['model']
        redfin_trace = input_data_dict['redfin_trace']
        validation_trace = input_data_dict['validation_trace']

        print(validation_trace['y'])
