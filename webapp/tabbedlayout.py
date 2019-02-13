#!/usr/bin/env python 

import dash_core_components as dcc
import dash_html_components as html 
import dash_table

from services import services

def build_layout(df):

    tabbed_layout = html.Div([

            html.Div(
                html.H3('Airvestment: Find your next investment property!')
                ),

            dcc.Tabs(
                id = 'tabs', 
                children = [
                    

                    dcc.Tab(
                        label = 'User Settings',
                        children = [
                            
                        html.H3('Please fill in the basic options below.'),

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

                            ]
                        ),

                    dcc.Tab(
                        label = 'Property Map',
                        children = [
                        
                            html.H5('Hover for coordinates and profit prediciton.'),
                            html.P('The color of each marker is indicative of the predicted profit.'),
                            dcc.Graph(
                                id = 'graph',
                                figure = {
                                    'data': [{'lat': df['latitude'], 'lon': df['longitude'], 'type': 'scattermapbox', 'text': df['profit']}],
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

                            ]
                        ), # End of tab 1 

                dcc.Tab(
                    label = 'Property List',
                    children = [
                        html.Div([
                                dcc.Interval(id='table-update',interval = 1000), 
                                dash_table.DataTable(
                                    id = 'table', 
                                    columns = [{'name':col, 'id':col} for col in services['table_cols']],
                                    data = services['data'][services['table_cols']].to_dict('records')
                                    )
                                ])
                        ]
                    )
                    ]) # End of dcc.Tabs()
            ])
        
    return tabbed_layout 
