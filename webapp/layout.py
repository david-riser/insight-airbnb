#!/usr/bin/env python 

import dash_core_components as dcc
import dash_html_components as html 
import dash_table

TABLE_COLS = ['bedrooms', 'bathrooms', 'URL', 'profit']

def build_layout(df):
    
    layout = html.Div([
            html.Div([
                    html.Div([
                        html.H3('Airvestment: Find your next investment property!'),
                        
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
                                        'style' : 'light'
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
                            type = 'text',
                            inputmode = 'numeric'
                            ),

                        dcc.Input(
                            id = 'max_price_input',
                            placeholder = 'Max. Price',
                            type = 'text',
                            inputmode = 'numeric'
                            ),

                        dcc.Input(
                            id = 'down_payment_input',
                            placeholder = 'Down Payment',
                            type = 'text',
                            inputmode = 'numeric' 
                            ),

                        dcc.Input(
                            id = 'loan_rate_input',
                            placeholder = 'Loan Rate',
                            type = 'text',
                            inputmode = 'numeric' 
                            ),

                        dcc.Input(
                            id = 'loan_term_input',
                            placeholder = 'Loan Term (Years)',
                            type = 'text',
                            inputmode = 'numeric' 
                            )

                        ], className="four columns"),
                ], className = 'row'),

            html.Div([
                    dash_table.DataTable(
                        id = 'table', 
                        columns = [{'name':col, 'id':col} for col in TABLE_COLS],
                        data = df[TABLE_COLS].to_dict('rows')
                        )
                    ], className = 'row')

        ])
    
    return layout 
