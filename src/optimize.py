#!/usr/bin/env python 

''' 

File:   optimize.py 
Author: David Riser
Date:   Feb. 8, 2019

This file is used to optimize the parameters of 
my model, including those related to feature 
creation.

'''

import numpy as np 
import pandas as pd 

from clean import create_clusters, create_crime_kde 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.externals import joblib 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm 

if __name__ == '__main__':

    # Setup 
    parameter_ranges = {
        'n_neighborhoods' : [3, 12],
        'kernel_bandwidth' : [0.0001, 0.01],
        'n_estimators' : [10, 150],
        'max_depth' : [3, 12]
        }

    parameter_settings = {
        'kernel_type' : ['exponential', 'gaussian', 'tophat']
        }

    number_random_trials = 1000

    # Load data and important things 
    dataset = pd.read_csv('./data/processed/airbnb.csv')
    dataset['log_price'] = np.log(dataset['price'].values)

    # Get training and testing
    features = ['bedrooms', 'bathrooms', 'latitude', 
                'longitude', 'crime_index', 'cluster_index',
                'mbta_1', 'mbta_2', 'mbta_3']

    # Add distance features
    for col in dataset.columns:
        if 'dist' in col:
            features.append(col)

    # Load the scaling transformation 
    transformer = joblib.load('./models/standard_scaler.pkl')

    # Generate some parameters 
    n_neighborhoods = np.random.randint(
        parameter_ranges['n_neighborhoods'][0],
        parameter_ranges['n_neighborhoods'][1],
        number_random_trials
        )

    kernel_bandwidth = np.random.uniform(
        parameter_ranges['kernel_bandwidth'][0],
        parameter_ranges['kernel_bandwidth'][1],
        number_random_trials
        )

    n_estimators = np.random.randint(
        parameter_ranges['n_estimators'][0],
        parameter_ranges['n_estimators'][1],
        number_random_trials
        )

    max_depth = np.random.randint(
        parameter_ranges['max_depth'][0],
        parameter_ranges['max_depth'][1],
        number_random_trials
        )

    kernel_type = np.random.choice(parameter_settings['kernel_type'], number_random_trials)
    random_seed = 1234 

    # Setup output dict for results
    output_data_dict = {}
    output_data_dict['n_neighborhoods'] = []
    output_data_dict['n_estimators'] = []
    output_data_dict['kernel_bandwidth'] = []
    output_data_dict['kernel_type'] = []
    output_data_dict['max_depth'] = []
    output_data_dict['train_score'] = []
    output_data_dict['test_score'] = []

    # Begin checking these
    for trial_index in tqdm(range(number_random_trials)):
        
        # Feature level stuff 
        create_clusters(dataset, redfin_data = None, k = n_neighborhoods[trial_index])
        create_crime_kde(
            dataset, 
            kernel_type = kernel_type[trial_index], 
            bandwidth = kernel_bandwidth[trial_index]
            )

        # Model stuff 
        model = RandomForestRegressor(
            n_estimators = n_estimators[trial_index],
            max_depth = max_depth[trial_index]
            )

        # This wastes time but it's quite easy compared to getting the 
        # correct columns to update with the information from clusters 
        # and crime.  This can be done to speed things up later. 
        x_train, x_test, y_train, y_test = train_test_split(dataset[features], dataset['log_price'], random_state = random_seed)
        x_train = transformer.transform(x_train)
        x_test = transformer.transform(x_test)

        # Train the model 
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_train_pred = model.predict(x_train)

        test_score = np.median(np.abs(np.exp(y_pred) - np.exp(y_test)))
        train_score = np.median(np.abs(np.exp(y_train_pred) - np.exp(y_train)))

        output_data_dict['n_neighborhoods'].append(n_neighborhoods[trial_index])
        output_data_dict['max_depth'].append(max_depth[trial_index])
        output_data_dict['n_estimators'].append(n_estimators[trial_index])
        output_data_dict['kernel_type'].append(kernel_type[trial_index])
        output_data_dict['kernel_bandwidth'].append(kernel_bandwidth[trial_index])
        output_data_dict['test_score'].append(test_score)
        output_data_dict['train_score'].append(train_score)        
        

    opt_param_results = pd.DataFrame(output_data_dict)
    opt_param_results.sort_values('test_score', inplace = True, ascending = True)
    
    # Run Final Model
        
    # Feature level stuff 
    create_clusters(dataset, redfin_data = None, k = n_neighborhoods[trial_index])
    create_crime_kde(
        dataset, 
        kernel_type = opt_param_results['kernel_type'].values[0], 
        bandwidth = opt_param_results['kernel_bandwidth'].values[0]
        )
    
    # Model stuff 
    model = RandomForestRegressor(
        n_estimators = opt_param_results['n_estimators'].values[0],
        max_depth = opt_param_results['max_depth'].values[0]
        )
    
    # This wastes time but it's quite easy compared to getting the 
    # correct columns to update with the information from clusters 
    # and crime.  This can be done to speed things up later. 
    x_train, x_test, y_train, y_test = train_test_split(dataset[features], dataset['log_price'], random_state = random_seed)
    x_train = transformer.transform(x_train)
    x_test = transformer.transform(x_test)

    # Train the model 
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    
    test_score = np.median(np.abs(np.exp(y_pred) - np.exp(y_test)))
    train_score = np.median(np.abs(np.exp(y_train_pred) - np.exp(y_train)))
    
    opt_param_results.to_csv('./data/metrics/optimize.csv', index = False)

    joblib.dump(model, './models/random_forest_optimized.pkl')

