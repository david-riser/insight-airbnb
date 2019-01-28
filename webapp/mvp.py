#!/usr/bin/env python 

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd

MAPBOX_ACCESS_TOKEN = 'pk.eyJ1IjoiZG1yaXNlciIsImEiOiJjanJmN3E4eW4yN283NDNwZDV6cWowMXNqIn0.oHNXyMiqEAgbyIrmprW2yA'

app = dash.Dash(__name__)

df = pd.read_csv('../data/predictions.csv')

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

app.layout = html.Div(
    children=[
        html.H4(children = 'Airvestments: Find your next real estate investment!'),

        dcc.Dropdown(
            id = 'bedrooms-dropdown', 
            options=[
                {'label': i, 'value': i} for i in df['BEDS'].unique()
                ], 
            multi=True, 
            placeholder = 'bedrooms...'),

        dcc.Dropdown(
            id = 'bathrooms-dropdown', 
            options=[
                {'label': i, 'value': i} for i in df['BATHS'].unique()
                ], 
            multi=True, 
            placeholder = 'bathrooms...'),

        html.Div(id='table-container')
])

@app.callback(
    dash.dependencies.Output('table-container', 'children'),
    [dash.dependencies.Input('bedrooms-dropdown', 'value'), 
     dash.dependencies.Input('bathrooms-dropdown', 'value')]
)
def display_table(bedrooms, bathrooms):

    dff = df[df['BEDS'] == bedrooms]
    dff = dff[dff['BATHS'] == bathrooms]
    dff.sort_values('airbnb_price', ascending = False, inplace = True)
    return generate_table(dff)

if __name__ == '__main__':
    app.run_server(debug=True)
