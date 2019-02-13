#!/usr/bin/env python 

import pandas as pd 

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
