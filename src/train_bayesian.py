#!/usr/bin/env python 

import numpy as np 
import pandas as pd 
import pymc3 as pm 
import pickle 

from sklearn.model_selection import train_test_split 
from theano import shared 

if __name__ == '__main__':

    # Settings 
    avg_occupancy = 0.7 
    days_per_month = 365.25 / 12.0

    # Load cleaned data 
    airbnb_data = pd.read_csv('./data/processed/airbnb.csv')
    redfin_data = pd.read_csv('./data/processed/redfin_boston.csv')

    # Spam the user.
    print('Loaded {} rows of training/testing data for AirBnb.'.format(len(airbnb_data)))
    print('Loaded {} rows of prediction data for redfin.'.format(len(redfin_data)))
    print('Columns {}'.format(airbnb_data.columns))
    print('Columns {}'.format(redfin_data.columns))

    # Get training and testing
    features = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'crime_index', 'cluster_index']

    # Add distance features
    for col in airbnb_data.columns:
        if 'dist' in col:
            features.append(col)

    # Dataset, log transformed price as the target. 
    x = airbnb_data[features].values
    y = np.log(airbnb_data['price'].values * avg_occupancy * days_per_month) 

    # Split into training and testing 
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    x_shared = shared(x_train)
    y_shared = shared(y_train)

    _, n_features = x_train.shape 

    print('Training data has shape {}'.format(x_train.shape))
    print('Testing data has shape {}'.format(x_test.shape))
    
    # Setup Bayesian Model 
    with pm.Model() as model:
        
        sigma = pm.HalfCauchy('sigma', beta = 10, testval = 1.0)
        intercept = pm.Normal('intercept', 0, sd = 20)
        
        # There should be a way to avoid hard coding the 
        # number of variables in the model.  The issue 
        # that I run into right now is how to perform the 
        # dot product used in the likelihood function.
        # This works, but isn't flexible.  It's also not 
        # really safe because the moment the input file changes
        # it's going to need to be updated.
        linear_coefs = []
        for index in range(n_features):
            linear_coefs.append(pm.Normal('coef_{}'.format(index), 0, sd = 20))
        
        # Define the ugly linear model 
        likelihood = pm.Normal(
            'y',
            mu = intercept + \
                linear_coefs[0]  * x_shared[:, 0] + \
                linear_coefs[1]  * x_shared[:, 1] + \
                linear_coefs[2]  * x_shared[:, 2] + \
                linear_coefs[3]  * x_shared[:, 3] + \
                linear_coefs[4]  * x_shared[:, 4] + \
                linear_coefs[5]  * x_shared[:, 5] + \
                linear_coefs[6]  * x_shared[:, 6] + \
                linear_coefs[7]  * x_shared[:, 7] + \
                linear_coefs[8]  * x_shared[:, 8] + \
                linear_coefs[9]  * x_shared[:, 9] + \
                linear_coefs[10] * x_shared[:, 10] + \
                linear_coefs[11] * x_shared[:, 11] + \
                linear_coefs[12] * x_shared[:, 12] + \
                linear_coefs[13] * x_shared[:, 13] + \
                linear_coefs[14] * x_shared[:, 14] + \
                linear_coefs[15] * x_shared[:, 15] + \
                linear_coefs[16] * x_shared[:, 16],
            sd = sigma,
            observed = y_shared
            )

        # Perform the sampling 
        trace = pm.sample(1000, cores = 2, tuning = 100)

        # Predict the testing sample 
        x_shared.set_value(x_test)
        y_shared.set_value(y_test)
        post_pred = pm.sample_posterior_predictive(trace, samples = 400)

        # Get predictions for the houses for sale. 
        x_new = redfin_data[features].values
        y_new = np.zeros(len(redfin_data))
        x_shared.set_value(x_new)
        y_shared.set_value(y_new)
        redfin_pred = pm.sample_posterior_predictive(trace, samples = 400)

        # Create a dictionary to save the output to a 
        # standard pickle file. 
        output_data_dict = {
            'trace' : trace, 
            'model' : model,
            'validation_trace' : post_pred,
            'redfin_trace' : redfin_pred,
            'x_test' : x_test,
            'y_test' : y_test
            }
        
        with open('./models/bayesian_model.pickle', 'wb') as buffer:
            pickle.dump(output_data_dict, buffer)

