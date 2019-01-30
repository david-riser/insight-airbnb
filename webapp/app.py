#!/usr/bin/env python 

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go

app = dash.Dash(__name__)
server = app.server 

# Prediction
df = pd.read_csv('./data/predictions/predictions.csv')

app.layout = html.Div([
        html.Div([
                html.Div([
                        html.H3('Airvestments: Find your next investment property!'),
                        
                        dcc.Graph(
                            id = 'graph',
                            figure = {
                                'data': [{'lat': df['latitude'], 'lon': df['longitude'], 'type': 'scattermapbox', 'text': df['airbnb_price']}],
                                'layout': {
                                    'mapbox': {
                                        'accesstoken': (
                                            'pk.eyJ1IjoiZG1yaXNlciIsImEiOiJjanJmN3E4eW4yN283NDNwZDV6cWowMXNqIn0.oHNXyMiqEAgbyIrmprW2yA'
                                            ),
                                        'center' : {
                                            'lat' : 42.3536, 
                                            'lon' : -71.0638 
                                            },
                                        'zoom' : 9.6,
                                        'style' : 'dark'
                                        },
                                    'margin': {
                                        'l': 0, 'r': 0, 'b': 0, 't': 0
                                        },
                                    }
                                }
                            )
                        
                        ], className = "eight columns"),
                
                html.Div([
                        html.H3('Enter Settings!'),

                        dcc.Input(
                            id = 'min_price_input',
                            placeholder = 'Min. Price',
                            type = 'text'
                            ),

                        dcc.Input(
                            id = 'max_price_input',
                            placeholder = 'Max. Price',
                            type = 'text' 
                            ),

                        dcc.Input(
                            id = 'down_payment_input',
                            placeholder = 'Down Payment',
                            type = 'text' 
                            ),

                        dcc.Input(
                            id = 'loan_rate_input',
                            placeholder = 'Loan Rate',
                            type = 'text' 
                            ),

                        dcc.Input(
                            id = 'loan_term_input',
                            placeholder = 'Loan Term (Years)',
                            type = 'text' 
                            )

                        ], className="four columns"),
                ], className="row"),

        ])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

def calculate_monthly_payment(loan_amount, rate = 0.05, years = 30):
    months = years * 12 
    c = rate / 12.0
    return loan_amount * ( c * (1 + c)**months ) / ( (1 + c)**months - 1 )
    
@app.callback(
    dash.dependencies.Output('graph', 'figure'), 
    [
     dash.dependencies.Input('min_price_input', 'value'),
     dash.dependencies.Input('max_price_input', 'value'),
     dash.dependencies.Input('down_payment_input', 'value'),
     dash.dependencies.Input('loan_rate_input', 'value'),
     dash.dependencies.Input('loan_term_input', 'value')
     ]
)
def update_map(min_price, max_price, down_payment, loan_rate, loan_term):

    # There is probably a better way to 
    # typecast these by using an input 
    # option or flag that automatically 
    # creates these as the correct type.
    down_payment = int(down_payment)
    loan_rate = float(loan_rate)
    loan_term = int(loan_term)
    
    dataset = df[df['PRICE'] > int(min_price)]
    dataset = dataset[dataset['PRICE'] < int(max_price)]

    dataset['monthly_payment'] = dataset['PRICE'].apply(lambda x: calculate_monthly_payment(x - down_payment, loan_rate, loan_term))

    occupancy = 0.7
    dataset['profit'] = occupancy * dataset['airbnb_price'] * (365.25 / 12.0) - dataset['monthly_payment'] 

    return {

        'data': [
            {
                'lat': dataset['latitude'], 
                'lon': dataset['longitude'], 
                'type': 'scattermapbox', 
                'text': dataset['profit']
                }
            ],

        'layout': {
            'mapbox': {
                'accesstoken': (
                    'pk.eyJ1IjoiZG1yaXNlciIsImEiOiJjanJmN3E4eW4yN283NDNwZDV6cWowMXNqIn0.oHNXyMiqEAgbyIrmprW2yA'
                    ),
                'center' : {
                    'lat' : 42.3536, 
                    'lon' : -71.0638 
                    },
                'zoom' : 9.6,
                'style' : 'dark'
                },
            'margin': {
                'l': 0, 'r': 0, 'b': 0, 't': 0
                },
            }
        }

if __name__ == '__main__':
    app.run_server(debug = True, port = 5678)
    
