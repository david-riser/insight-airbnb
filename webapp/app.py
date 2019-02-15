#!/usr/bin/env python 

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import utils 

from services import services 
from tabbedlayout import build_layout

app = dash.Dash(__name__)

# Although this is not used in the 
# code, it seems to be important for the 
# Heroku app to work.  This is the Flask
# server that underlies the dash app. 
server = app.server 

# Load data and apply transformation to have required 
# columns for this website. 
df = pd.read_csv('./data/predictions/predictions.csv')
utils.add_profit_to_dataframe(
    dataset = df, 
    down_payment = 0, 
    loan_rate = 0.01, 
    loan_term = 30
    )

services['data'] = df
services['data_subset'] = df

# Setup the document layout 
app.layout = build_layout(df) 

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
    
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
    
    dataset = df[df['price'] > int(min_price)]
    dataset = dataset[dataset['price'] < int(max_price)]

    # This should run first too idk how. 
    utils.add_profit_to_dataframe(dataset, down_payment, loan_rate, loan_term)

    # Add to my global data storage area. 
    services['data_subset'] = dataset

    return {

        'data': [
            {
                'lat': dataset['latitude'], 
                'lon': dataset['longitude'], 
                'type': 'scattermapbox', 
                'text': dataset['profit'],
                'marker' : {
                    'line' : {'width' : 1},
                    'color' : dataset['profit']
                    }
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
                'style' : 'light'
                },
            'margin': {
                'l': 0, 'r': 0, 'b': 0, 't': 0
                },
            }
        }

@app.callback(
    dash.dependencies.Output('table', 'data'),
    [
     dash.dependencies.Input('table-update', 'n_intervals'),
     ]
)
def update_table(n_intervals):
    return services['data_subset'][services['table_cols']].to_dict('records')

if __name__ == '__main__':
    app.run_server(debug = True, port = 5678)
    
