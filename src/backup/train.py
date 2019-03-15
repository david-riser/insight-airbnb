#!/usr/bin/env python 

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def r2_score(y_pred, y_true):
    return 1 - np.sum((y_pred - y_true)**2) / np.sum(y_true**2)

def train_kfold(x, y, k = 5, model_name = 'unnamed', 
                builder = RandomForestRegressor, params = None, log_transform = False):
    ''' Train and CV a model built by builder. '''

    # Setup 5 fold CV
    kf = KFold(n_splits = k)

    # Out of fold predictions to 
    # study performance
    oof_predictions = np.zeros(len(y))

    train_metric = np.zeros(k)
    valid_metric = np.zeros(k)

    fold_index = 0
    for train_index, valid_index in kf.split(x):

        print('[{}] Starting fold {}'.format(model_name, fold_index))

        # Setup simple model
        model = builder(**params)

        if log_transform:
            model.fit(x[train_index], np.log(y[train_index]))
        else:
            model.fit(x[train_index], y[train_index])

        if log_transform:
            y_pred = np.exp(model.predict(x[valid_index]))
        else:
            y_pred = model.predict(x[valid_index])

        oof_predictions[valid_index] = y_pred 

        # Metric report
        y_train_pred = model.predict(x[train_index])
        train_rmse = np.sqrt(mean_squared_error(np.exp(y_train_pred), np.exp(y[train_index])))
        valid_rmse = np.sqrt(mean_squared_error(np.exp(y_pred), np.exp(y[valid_index])))
        
        train_metric[fold_index] = train_rmse
        valid_metric[fold_index] = valid_rmse

        # Save Model
        output_model_name = './models/{}_{}.joblib'.format(model_name, fold_index)
        print('[{}] Saving model to {}.'.format(model_name, output_model_name))

        joblib.dump(model, output_model_name)
 
        if model_name == 'elastic_net':
            print('[{}] '.format(model_name), model.coef_)
        elif model_name == 'random_forest':
            print('[{}] '.format(model_name), model.feature_importances_)
            
            plt.clf()
            plt.barh(features, model.feature_importances_, edgecolor = 'k')
            plt.savefig('./image/rf_feature_importance_{}.png'.format(fold_index), bbox_inches = 'tight')

        fold_index += 1

    train_axis_label = ['Train RMSE (Fold {})'.format(ind) for ind in range(k)]
    valid_axis_label = ['Validation RMSE (Fold {})'.format(ind) for ind in range(k)]

    plt.clf()
    plt.barh(
        train_axis_label,
        train_metric,
        edgecolor = 'k'
             )
    plt.savefig('./image/{}_training_metric.png'.format(model_name), bbox_inches = 'tight')

    plt.clf()
    plt.barh(
        valid_axis_label,
        valid_metric,
        edgecolor = 'k'
             )
    plt.savefig('./image/{}_validation_metric.png'.format(model_name), bbox_inches = 'tight')

    return oof_predictions 

def train(x, x_simple, y, model_type, builder = RandomForestRegressor, params = None):
    ''' This function trains the model a few times and then 
    returns the metrics for each setup. '''

    # build schema of output df 
    output_data_dict = {}
    output_data_dict['model_type'] = []
    output_data_dict['train_rmse'] = []
    output_data_dict['train_mape'] = []
    output_data_dict['valid_rmse'] = []    
    output_data_dict['valid_mape'] = []
    output_data_dict['feature_type'] = []
    output_data_dict['log_transform'] = []
    
    # split data 
    x_train, x_test, y_train, y_test, x_simple_train, x_simple_test = train_test_split(x, y, x_simple)

    # train simple 
    output_data_dict['model_type'].append(model_type)
    output_data_dict['feature_type'].append('simple')
    output_data_dict['log_transform'].append(False)

    rf_simple = builder(**params)
    rf_simple.fit(x_simple_train, y_train)
    y_pred = rf_simple.predict(x_simple_test)
    y_train_pred = rf_simple.predict(x_simple_train)

    rmse, mape = calculate_metrics(y_train_pred, y_train)
    output_data_dict['train_rmse'].append(rmse)
    output_data_dict['train_mape'].append(mape)

    rmse, mape = calculate_metrics(y_pred, y_test)
    output_data_dict['valid_rmse'].append(rmse)
    output_data_dict['valid_mape'].append(mape)

    # train simple log
    output_data_dict['model_type'].append(model_type)
    output_data_dict['feature_type'].append('simple')
    output_data_dict['log_transform'].append(True)

    rf_simple_log = builder(**params)
    rf_simple_log.fit(x_simple_train, np.log(y_train))
    y_pred = np.exp(rf_simple_log.predict(x_simple_test))
    y_train_pred = np.exp(rf_simple_log.predict(x_simple_train))

    rmse, mape = calculate_metrics(y_train_pred, y_train)
    output_data_dict['train_rmse'].append(rmse)
    output_data_dict['train_mape'].append(mape)

    rmse, mape = calculate_metrics(y_pred, y_test)
    output_data_dict['valid_rmse'].append(rmse)
    output_data_dict['valid_mape'].append(mape)

    # train full
    output_data_dict['model_type'].append(model_type)
    output_data_dict['feature_type'].append('full')
    output_data_dict['log_transform'].append(False)

    rf = builder(**params)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    y_train_pred = rf.predict(x_train)

    rmse, mape = calculate_metrics(y_train_pred, y_train)
    output_data_dict['train_rmse'].append(rmse)
    output_data_dict['train_mape'].append(mape)

    rmse, mape = calculate_metrics(y_pred, y_test)
    output_data_dict['valid_rmse'].append(rmse)
    output_data_dict['valid_mape'].append(mape)

    # train full log
    output_data_dict['model_type'].append(model_type)
    output_data_dict['feature_type'].append('full')
    output_data_dict['log_transform'].append(True)

    rf_log = builder(**params)
    rf_log.fit(x_train, np.log(y_train))
    y_pred = np.exp(rf_log.predict(x_test))
    y_train_pred = np.exp(rf_log.predict(x_train))

    rmse, mape = calculate_metrics(y_train_pred, y_train)
    output_data_dict['train_rmse'].append(rmse)
    output_data_dict['train_mape'].append(mape)

    rmse, mape = calculate_metrics(y_pred, y_test)
    output_data_dict['valid_rmse'].append(rmse)
    output_data_dict['valid_mape'].append(mape)

    joblib.dump(rf_log, './models/{}_log_trans.pkl'.format(model_type))
    
    return pd.DataFrame(output_data_dict)

def calculate_metrics(y_pred, y_test):
    rmse = np.sqrt(np.mean( (y_pred - y_test)**2 ) )
    mape = np.median(np.abs(y_pred - y_test))
    return rmse, mape

if __name__ == '__main__':

    # Load clean datasets
    airbnb_data = pd.read_csv('./data/processed/airbnb.csv')
    redfin_data = pd.read_csv('./data/processed/redfin_boston.csv')

    # Get training and testing
    base_features = ['bedrooms', 'bathrooms', 'latitude', 'longitude']
    features = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'crime_index', 'cluster_index', 
                'mbta_1', 'mbta_2', 'mbta_3']

    # Add distance features
    for col in airbnb_data.columns:
        if 'dist' in col:
            features.append(col)

    x_simple = airbnb_data[base_features].values
    x = airbnb_data[features].values
    y = airbnb_data['price'].values

    # Scale 
    ss = StandardScaler() 
    x = ss.fit_transform(x)
    joblib.dump(ss, './models/standard_scaler.pkl')
    

    simple_ss = StandardScaler()
    x_simple = simple_ss.fit_transform(x_simple)

    rf_params = {
        'n_jobs' : -1,
        'n_estimators' : 75,
        'max_depth' : 6
    }

    elastic_net_params = {
        'fit_intercept' : True,
        'alpha' : 1.0
    }

    gb_params = {
        
    }

    df_rf = train(
        x = x, 
        x_simple = x_simple, 
        y = y, 
        model_type = 'random_forest', 
        builder = RandomForestRegressor,
        params = rf_params
        )

    df_rf.to_csv('./data/metrics/rf.csv', index = False)

    df_gb = train(
        x = x, 
        x_simple = x_simple, 
        y = y, 
        model_type = 'random_forest', 
        builder = GradientBoostingRegressor,
        params = gb_params
        )

    df_gb.to_csv('./data/metrics/gb.csv', index = False)

    df_lin = train(
        x = x, 
        x_simple = x_simple, 
        y = y, 
        model_type = 'elastic_net', 
        builder = ElasticNet,
        params = elastic_net_params
        )
    df_lin.to_csv('./data/metrics/elastic_net.csv', index = False)

    '''
    oof_preds = train_kfold(
        x = x_simple,
        y = y,
        k = 5,
        model_name = 'random_forest_simple',
        builder = RandomForestRegressor,
        params = rf_params
    )
    # Save this to analyze later 
    airbnb_data['oof_pred'] = oof_preds
    airbnb_data.to_csv('./data/predictions/oof_random_forest_simple_features.csv', index = False)

    oof_preds = train_kfold(
        x = x,
        y = y,
        k = 5,
        model_name = 'random_forest',
        builder = RandomForestRegressor,
        params = rf_params
    )

    # Save this to analyze later 
    airbnb_data['oof_pred'] = oof_preds
    airbnb_data.to_csv('./data/predictions/oof_random_forest.csv', index = False)


    # For a linear model let's remove lat/long 
    features.remove('latitude')
    features.remove('longitude')
    x = airbnb_data[features].values

    print('Moving to elastic_net w/ features: ', features)
    oof_preds = train_kfold(
        x = x,
        y = y,
        k = 5,
        model_name = 'elastic_net',
        builder = ElasticNet,
        params = elastic_net_params
    )

    # Save this to analyze later 
    airbnb_data['oof_pred'] = oof_preds
    airbnb_data.to_csv('./data/predictions/oof_elastic_net.csv', index = False)

    oof_preds = train_kfold(
        x = x_simple,
        y = y,
        k = 5,
        model_name = 'elastic_net_simple',
        builder = ElasticNet,
        params = elastic_net_params
    )

    # Save this to analyze later 
    airbnb_data['oof_pred'] = oof_preds
    airbnb_data.to_csv('./data/predictions/oof_elastic_net_simple_features.csv', index = False)

    '''
