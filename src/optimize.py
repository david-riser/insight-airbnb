#!/usr/bin/env python 

''' 

File:   optimize.py 
Author: David Riser
Date:   Feb. 8, 2019

This file is used to optimize the parameters of 
my model, including those related to feature 
creation.

'''

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

from clean import create_clusters, create_crime_kde 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.externals import joblib 
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm 

def build_feature_list(dataset):

    # Get training and testing
    features = ['bedrooms', 'bathrooms', 'latitude', 
                'longitude', 'crime_index', 'cluster_index',
                'mbta_1', 'mbta_2', 'mbta_3']

    # Add distance features
    for col in dataset.columns:
        if 'dist' in col:
            features.append(col)

    return features 

def plot_feature_importance(model, features, image_path):
    ''' Save a plot of feature importance. 
    
    Inputs: 
    model      -> input sklearn tree based model
    features   -> list of features used in training
    image_path -> the save directory 

    ''' 

    feat_names = {
        'dist_0' : 'Dist. Back Bay T Station',
        'dist_1' : 'Dist. Harvard Square T Station',
        'dist_2' : 'Dist. Museum of Science',
        'dist_3' : 'Dist. Faneuil Hall',
        'dist_4' : 'Dist. Copley Square',
        'dist_5' : 'Dist. Museum of Fine Art',
        'dist_6' : 'Dist. Freedom Trail (start)',
        'dist_7' : 'Dist. Sam Adams',
        'dist_8' : 'Dist. Toyota of Braintree',
        'crime_index' : 'Crime Index',
        'dist_mbta_1' : 'Dist. Closest T Station',
        'dist_mbta_2' : 'Dist. 2nd Closest T Station',
        'dist_mbta_3' : 'Dist. 3rd Closest T Station',
        'mbta_1' : 'Closest T Station Code',
        'mbta_2' : '2nd Closest T Station Code',
        'mbta_3' : '3rd Closest T Station Code',
        'cluster_index' : 'Neighborhood Cluster'
    }

    plotted_features = []
    for feature in features:
        if feature in feat_names.keys():
            plotted_features.append(feat_names[feature])
        else:
            plotted_features.append(feature)

    # Sort by size decreasing 
    indices = np.argsort(model.feature_importances_)

    plt.rc('font', size = 18)
    plt.rc('font', family = 'serif')
    plt.figure(figsize = (9, 16))
    plt.barh(
        [plotted_features[idx] for idx in indices], 
        [model.feature_importances_[idx] for idx in indices],
        edgecolor = 'k'
        )
    plt.xlabel('Feature Importance')
    plt.savefig('{}/optimized_feature_importances.png'.format(image_path), bbox_inches = 'tight')
    
def load_dataset(data_dir):
    dataset = pd.read_csv('{}/airbnb.csv'.format(data_dir))
    dataset['log_price'] = np.log(dataset['price'].values)
    return dataset

def train_kfold(x, y, k = 5, builder = RandomForestRegressor, params = None):
    ''' Train and CV a model built by builder. '''

    # Setup 5 fold CV
    kf = KFold(n_splits = k)

    train_metric = np.zeros(k)
    valid_metric = np.zeros(k)

    fold_index = 0 
    for train_index, valid_index in kf.split(x):

        # Setup simple model
        model = builder(**params)
 
        model.fit(x[train_index], y[train_index])
        y_pred = model.predict(x[valid_index])

        # Metric report
        y_train_pred = model.predict(x[train_index])
        train_metric[fold_index] = np.median(np.abs(np.exp(y_train_pred) - np.exp(y[train_index])))
        valid_metric[fold_index] = np.median(np.abs(np.exp(y_pred) - np.exp(y[valid_index])))

        fold_index += 1

    return np.mean(train_metric), np.mean(valid_metric)

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

    # Constants 
    number_random_trials = 200
    random_seed = 1234 

    dataset = load_dataset('./data/processed')
    features = build_feature_list(dataset)

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

        x = dataset[features].values
        xp = transformer.transform(x)

        y = dataset['log_price'].values

        rf_params = {
            'n_estimators' : n_estimators[trial_index],
            'max_depth' : max_depth[trial_index],
            }

        train_score, test_score = train_kfold(x, y, 5, RandomForestRegressor, rf_params)

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

    # Save things for later. 
    opt_param_results.to_csv('./data/metrics/optimize.csv', index = False)
    joblib.dump(model, './models/random_forest_optimized.pkl')

    # Create the final feature importance plot.
    plot_feature_importance(model, features, './image')
