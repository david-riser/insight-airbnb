#!/usr/bin/env python

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
import utils

def build_layout(dataset, table_cols):
    tabbed_layout = html.Div([

        dcc.Store(id = 'memory-store'),

        html.Div(
            html.H3('Airvestment: Find your next investment property!')
        ),

        dcc.Tabs(
            id='tabs',
            children=[

                dcc.Tab(
                    label='User Settings',
                    children=[

                        html.H3('Please fill in the basic options below.'),

                        dcc.Input(
                            id='min_price_input',
                            placeholder='Min. Price',
                            type='text',
                            inputmode='numeric'
                        ),

                        dcc.Input(
                            id='max_price_input',
                            placeholder='Max. Price',
                            type='text',
                            inputmode='numeric'
                        ),

                        dcc.Input(
                            id='down_payment_input',
                            placeholder='Down Payment',
                            type='text',
                            inputmode='numeric'
                        ),

                        dcc.Input(
                            id='loan_rate_input',
                            placeholder='Loan Rate',
                            type='text',
                            inputmode='numeric'
                        ),

                        dcc.Input(
                            id='loan_term_input',
                            placeholder='Loan Term (Years)',
                            type='text',
                            inputmode='numeric'
                        ),

                        html.Div([
                            html.A(
                                'Demo Slides',
                                href='https://docs.google.com/presentation/d/11DrCeIs5ouWqQDapwz1ZXGd7EIGWaev15KhjRCsY5qU/edit?usp=sharing')
                        ])
                    ]
                ),

                dcc.Tab(
                    label='Property Map',
                    children=[

                        html.H5('Hover for coordinates and profit prediciton.'),
                        html.P('The color of each marker is indicative of the predicted profit.'),
                        dcc.Graph(
                            id='graph',
                            figure={
                                'data': [{'lat': dataset['latitude'],
                                          'lon': dataset['longitude'], 'type': 'scattermapbox',
                                          'text': dataset['profit']}],
                                'layout': {
                                    'mapbox': {
                                        'accesstoken': (
                                            'pk.eyJ1IjoiZG1yaXNlciIsImEiOiJjanJmN3E4eW4yN283NDNwZDV6cWowMXNqIn0.oHNXyMiqEAgbyIrmprW2yA'
                                        ),
                                        'center': {
                                            'lat': 42.3536,
                                            'lon': -71.0638
                                        },
                                        'zoom': 9.6,
                                        'style': 'light'
                                    },
                                    'margin': {
                                        'l': 0, 'r': 0, 'b': 0, 't': 0
                                    },
                                }
                            }
                        )
                    ]
                ),  # End of tab 1

                dcc.Tab(
                    label='Property List',
                    children=[
                        html.Div([
                            dcc.Interval(id='table-update', interval=1000),
                            dash_table.DataTable(
                                id='table',
                                columns = [{'name': col, 'id': col} for col in table_cols],
                                data = dataset[table_cols].to_dict('records')
                            )
                        ])
                    ]
                )
            ])  # End of dcc.Tabs()
    ])

    return tabbed_layout

app = dash.Dash(__name__)

# Although this is not used in the
# code, it seems to be important for the
# Heroku app to work.  This is the Flask
# server that underlies the dash app.
server = app.server

# Load data and apply transformation to have required
# columns for this website.
data = pd.read_csv('../data/predictions/predictions.csv')
data = utils.add_profit_to_dataframe(
    dataset = data,
    down_payment = 0,
    loan_rate = 0.01,
    loan_term = 30
    )

# Define table structure and build the app layout.
table_cols = ['bedrooms', 'bathrooms', 'URL', 'profit']
app.layout = build_layout(data, table_cols)

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

@app.callback(
    dash.dependencies.Output('memory-store', 'data'),
    [
        dash.dependencies.Input('min_price_input', 'value'),
        dash.dependencies.Input('max_price_input', 'value'),
        dash.dependencies.Input('down_payment_input', 'value'),
        dash.dependencies.Input('loan_rate_input', 'value'),
        dash.dependencies.Input('loan_term_input', 'value')
    ]
)
def filter_properties(min_price, max_price, down_payment, loan_rate, loan_term):
    ''' Filter the properties according to the settings supplied
    by the user and store them in the memory store object.
    '''

    filtered_data = data.query('price >= {} and price <= {}'.format(min_price, max_price))
    filtered_data = utils.add_profit_to_dataframe(
        dataset = filtered_data,
        down_payment = down_payment,
        loan_rate = loan_rate,
        loan_term = loan_term
    )

    return filtered_data.to_dict('rows')


@app.callback(
    dash.dependencies.Output('table', 'data'),
    [
        dash.dependencies.Input('memory-store', 'data')
    ]
)
def update_data_table(data):
    ''' Construct the data-table if the data exists. '''

    if not data:
        raise dash.exceptions.PreventUpdate
    else:
        return data

@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [
        dash.dependencies.Input('memory-store', 'data')
    ]
)
def update_map(data):

    if not data:
        raise dash.exceptions.PreventUpdate
    else:

        dataset = pd.DataFrame(data)
        print('{} entries remain in the dataset after loading into update_map function.'.format(len(dataset)))

        return {
            'data': [
            {
                'lat': dataset['latitude'],
                'lon': dataset['longitude'],
                'type': 'scattermapbox',
                'text': dataset['profit'],
                'marker': {
                    'line': {'width': 1},
                    'color': dataset['profit']
                }
            }
        ],

        'layout': {
            'mapbox': {
                'accesstoken': (
                    'pk.eyJ1IjoiZG1yaXNlciIsImEiOiJjanJmN3E4eW4yN283NDNwZDV6cWowMXNqIn0.oHNXyMiqEAgbyIrmprW2yA'
                ),
                'center': {
                    'lat': 42.3536,
                    'lon': -71.0638
                },
                'zoom': 9.6,
                'style': 'light'
            },
            'margin': {
                'l': 0, 'r': 0, 'b': 0, 't': 0
            },
        }
    }


if __name__ == '__main__':
    app.run_server(debug = True, port = 5678)



