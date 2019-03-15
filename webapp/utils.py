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

def typecast_to_int(variable):
    ''' Ensure the variable is an integer. '''

    # Just don't process None
    if not variable:
        return variable

    if not isinstance(variable, int):
        return int(variable)
    else:
        return variable

def typecast_to_float(variable):
    ''' Ensure the variable is an integer. '''

    if not variable:
        return variable

    if not isinstance(variable, float):
        return float(variable)
    else:
        return variable

def add_profit_to_dataframe(dataset, down_payment, loan_rate, loan_term, occupancy = 0.7):
    ''' Use simple calculation to add profit to df. '''

    # There should be some way to specify variable types in
    # the different input boxes.
    down_payment = typecast_to_int(down_payment)
    loan_rate = typecast_to_float(loan_rate)
    loan_term = typecast_to_float(loan_term)

    dataset['monthly_payment'] = dataset['price'].apply(lambda x: calculate_monthly_payment(x - down_payment, loan_rate, loan_term))
    dataset['profit'] = occupancy * dataset['airbnb_price'] * (365.25 / 12.0) - dataset['monthly_payment']
    return dataset.sort_values('profit', ascending = False)
