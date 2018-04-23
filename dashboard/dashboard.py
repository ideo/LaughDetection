import dash_core_components as dcc
import dash_html_components as html
import dash
import plotly.graph_objs as go



import pandas as pd

threshold = 0.6
laugh_quota = 100

d2 = pd.read_csv('Data/test.csv', names=['time', 'score', 'vol'])

d2['laugh'] = d2['score'] > threshold

d2['time'] = pd.to_datetime(d2['time'])
start = pd.Timestamp('2018-02-28 11:14:38')
ts = pd.date_range(start, periods=d2.shape[0], freq='3S')
d2['ts'] = ts
d2 = d2.set_index('ts')
today = d2.loc[d2.index > '2018-03-07 00:00:00']


d2_window = today[['laugh', 'vol']].rolling(window=20).sum()
time_agg = d2.groupby(pd.TimeGrouper('1D')).sum()

total_laughs = today['laugh'].sum()
remaining_laughs = laugh_quota - total_laughs

donut_plot = dcc.Graph(
        id='donut-plot',
        figure={
            'data': [
                {
                  "values": [total_laughs, remaining_laughs],
                  "labels": ['laughs today', 'laughs left'],
                  "marker": {"colors": ['#3E83F2', '#ADB9CC']},
                  "direction": 'clockwise',
                  "sort": False,
                  "domain": {"x": [0, 1]},
                  "hoverinfo": "none",
                  "textinfo": "none",
                  "hole": .6,
                  "type": "pie",
                  "showlegend": False
                }
                ],
            'layout': {
                'title': 'Laugh Quota',
                'margin': {'l':10, 'r':10, 'b':10, 't':60},
                "annotations": [
                    {
                        "font": {
                            "size": 20
                        },
                        "showarrow": False,
                        "text": f'{total_laughs} Laughs',
                        "x": 0.5,
                        "y": 0.5,
                        "xanchor": "center",
                        "yanchor": "center"
                    },
                ]
            }
        }
    )


line_plot = dcc.Graph(
        id='line-plot',
        figure={
            'data': [
                {'x': d2_window.index, 'y': d2_window['laugh'], 'type': 'scatter'},
            ],
            'layout': {
                'title': 'Laugh-o-Meter'
            }
        }
    )

bar_plot = dcc.Graph(
        id='bar-plot',
        figure={
            'data': [
                {'x': time_agg.index, 'y': time_agg['laugh'], 'type': 'bar'},
            ],
            'layout': {
                'title': 'Daily Laughs'
            }
        }
    )


app = dash.Dash()

app.layout = html.Div(children=[
    html.Div(
        children=[donut_plot],
        style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}
    ),
    html.Div(
            children=[line_plot,bar_plot],
            style={'width': '70%', 'display': 'inline-block'}
    )],
    style={'width': '100%', 'display': 'inline-block', 'align': 'center'}
)


if __name__ == '__main__':
    app.run_server()
