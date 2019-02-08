#!/usr/bin/env python 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.externals import joblib

def calculate_monthly_payment(loan_amount, rate = 0.05, years = 30):

    months = years * 12

    # Default case to 30 years
    if months <= 0:
        months = 30 * 12

    # Default case to 1% rate
    if rate <= 0:
        rate = 0.01

    c = rate / 12.0
    return loan_amount * ( c * (1 + c)**months ) / ( (1 + c)**months - 1 )

def add_profit_to_dataframe(dataset, down_payment, loan_rate, loan_term, occupancy = 0.7):
    dataset['monthly_payment'] = dataset['price'].apply(lambda x: calculate_monthly_payment(x - down_payment, loan_rate, loan_term))
    dataset['profit'] = occupancy * dataset['airbnb_price'] * (365.25 / 12.0) - dataset['monthly_payment']
    dataset.sort_values('profit', ascending = False, inplace = True)

def kill_name_of_horrible_url_column(data):
    
    # Kill the horrible name 
    kill_index = -1
    for col_index, col in enumerate(data.columns):
        if 'URL' in col:
            kill_index = col_index

    if kill_index > -1:
        new_cols = list(data.columns)
        new_cols[kill_index] = 'URL'
        data.columns = new_cols

if __name__ == '__main__':

    # Input data 
    data = pd.read_csv('./data/processed/redfin_boston.csv')
    model_name = 'random_forest'
    model = joblib.load('./models/{}_optimized.pkl'.format(model_name))

    kill_name_of_horrible_url_column(data)

    # Predict these houses
    features = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'crime_index', 
                'cluster_index', 'mbta_1', 'mbta_2', 'mbta_3']
    keep_cols = ['bedrooms', 'bathrooms', 'latitude', 'longitude', 'crime_index', 'price', 'cluster_index', 'mbta_1', 'mbta_2', 'mbta_3']
    keep_cols.append('URL')

    for col in data.columns:
        if 'dist' in col:
            features.append(col)
            keep_cols.append(col)

    # Basic cleanup 
    data = data[keep_cols]
    data.dropna(how = 'any', inplace = True)

    # Standardization 
    ss = joblib.load('./models/standard_scaler.pkl')
    xp = ss.transform(data[features].values)

    data['airbnb_log_price'] = model.predict(xp)

    # I have to kill anything that comes out zero or negative
    # due to the nature of the transformation. 
    data = data[data['airbnb_log_price'] > 0]

    data['airbnb_price'] = np.exp(data['airbnb_log_price'])

    #    data['monthly_revenue'] = 0.7 * 365.25 / 12.0 * data['airbnb_price'].values

    add_profit_to_dataframe(data, down_payment = 0, loan_rate = 0.01, loan_term = 30, occupancy = 0.7)
    
    plt.hist(data['profit'].values, bins = 40, edgecolor = 'k')
    plt.show()

    profit_buckets = np.array([-np.inf, 0, 500, 1000, np.inf])
    profit_categories = ['loss', 'low', 'moderate', 'lucrative']
    
    indices = np.digitize(data['profit'].values, profit_buckets)
    print(indices)
    data['profit_category'] = [profit_categories[i - 1] for i in indices]
    

    # Save output 
    data.to_csv('./data/predictions/predictions.csv', index = False)
 
    print(data.sort_values('profit', ascending = False))
    
    # Offer some simple quantitative statement about 
    # the efficacy of the product. 
    print(data.groupby('profit_category').agg({'profit' : [np.mean, lambda x: len(x) / len(data)]}))
    print('Mean profit: {}'.format(data['profit'].mean()))
