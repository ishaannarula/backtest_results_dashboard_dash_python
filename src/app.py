import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash import dash_table
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Format, Scheme

import holoviews as hv
hv.extension('bokeh')

import warnings
warnings.filterwarnings('ignore')
import datetime as dt
# import os
# os.chdir('/Users/ishaan/Library/CloudStorage/GoogleDrive-ishaan.narula1@gmail.com/My Drive/Personal Development/GitHub/backtest_results_dashboard_dash_python')
import backtest_functions as btfunc


# Run the backtesting pipeline

# Create dataframes
initTicker = 'QQQ'
initSma = 50
initStartingCapital = 100000


(
    tickerClose_dataClean,
    tickerClose_tradingIndic,
    tickerClose_exposureTrades,
    tickerClose_retsDaily,
    tickerClose_rets252d,
    tickerClose_retsAnn,
    # tickerClose_trades
) = btfunc.long_ma_short_short_back_test(
    add_ticker=initTicker,
    price_for_analysis='close',
    bt_start=dt.datetime(2014, 1, 1),
    bt_end=dt.datetime(2021, 12, 31),
    sma_number=initSma,
    starting_capital=initStartingCapital
)

# Calculate performance statistics
(
    performStat252d,
    performStatAnn,
    performStatCagr,
    # trades_df
) = btfunc.results(
    rets_daily=tickerClose_retsDaily,
    rets_252d=tickerClose_rets252d,
    rets_annual=tickerClose_retsAnn,
    # trades_df=tickerClose_trades

)

# Create Plotly charts
fig = btfunc.visualise_strategy_returns_plotly(df=tickerClose_retsDaily, ticker_Name=initTicker)

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define the layout

# Inputs section elements
ticker = [
    html.Div(
        [
            html.Label('Ticker',
                       style={'font-size': '14px',
                              'margin-bottom': '5px'
                       }
            ),
            dcc.Input(id='ticker',
                      type='text',
                      debounce=True,
                      placeholder='Enter ticker...',
                      value=initTicker,
                      required=True,
                      style={'font-size': '14px'}
            ),
        ],
        style={'margin-bottom': '15px'}
    )
]

startDate = [
    html.Div(
        [
            html.Label('Start date (yyyy/mm/dd)',
                       style={'font-size': '14px',
                              'margin-bottom': '5px'
                       }
            ),
            dcc.Input(id='start_date',
                      type='text',
                      debounce=True,
                      value=str(tickerClose_exposureTrades.index.min().date()),
                      required=True,
                      style={'font-size': '14px'}
            ),
        ],
        style={'margin-bottom': '15px'}
    )
]

endDate = [
    html.Div(
        [
            html.Label('End date (yyyy/mm/dd)',
                       style={'font-size': '14px',
                              'margin-bottom': '5px'
                       }
            ),
            dcc.Input(id='end_date',
                      type='text',
                      debounce=True,
                      value=str(tickerClose_exposureTrades.index.max().date()),
                      required=True,
                      style={'font-size': '14px'}
            ),
        ],
        style={'margin-bottom': '15px'}
    )
]

movingAvg = [
    html.Div(
        [
            html.Label('Moving average',
                       style={'font-size': '14px',
                              'margin-bottom': '5px'
                       }
            ),
            dcc.Input(id='sma',
                      type='number',
                      debounce=True,
                      placeholder='Enter a moving avg. number...',
                      value=initSma,
                      min=1,
                      max=500,
                      step=1
            ),
        ],
        style={'margin-bottom': '15px'}
    ),
]

# Dashboard tabs

# Charts tab
chartsTab = [
    dbc.Tab(
        dcc.Graph(id='my_graph', figure=fig),
        label='Interactive charts',
        tab_style={'margin-left': 'auto', 'margin-right': 'auto'}
    ),
]

# Results dataframe tab
resultsdfTab = [
    dbc.Tab(
        dash_table.DataTable(
            id='resultsdf',
            columns=[
                {
                    'name': i,
                    'id': i,
                    'type': 'numeric',
                    'format': Format(precision=4, scheme=Scheme.fixed)
                }

                for i in tickerClose_retsDaily.columns
            ],
            data=tickerClose_retsDaily.to_dict('records'),
            export_format='csv',
            export_headers='display',

            # Formatting options
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            column_selectable='multi',
            row_selectable='multi',
            page_action='native',
            style_cell={
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '14px',
                'padding': '8px',
                'textAlign': 'center'
            },
            style_header={
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'},
                {'if': {'row_index': 'even'}, 'backgroundColor': 'white'},
                {'if': {'column_id': 'date'}, 'textAlign': 'center', 'fontWeight': 'bold'}
            ],
            style_as_list_view=True
        ),
        label='Results dataframe',
        tab_style={'margin-left': 'auto', 'margin-right': 'auto'},
    )
]

# Rolling 252-day returns tab
roll252dRetsTab = [
    dbc.Tab(
        dash_table.DataTable(
            id='dfDashb252d',
            columns=[
                {
                    'name': i,
                    'id': i,
                    'type': 'numeric',
                    'format': Format(precision=4, scheme=Scheme.fixed)
                }

                for i in tickerClose_rets252d.columns
            ],
            data=tickerClose_rets252d.to_dict('records'),
            export_format='csv',
            export_headers='display',

            # Formatting options
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            column_selectable='multi',
            row_selectable='multi',
            page_action='native',
            style_cell={
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '14px',
                'padding': '8px',
                'textAlign': 'center',
            },
            style_header={
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'},
                {'if': {'row_index': 'even'}, 'backgroundColor': 'white'},
                {'if': {'column_id': 'date'}, 'textAlign': 'center', 'fontWeight': 'bold'}
            ],
            style_as_list_view=True
        ),
        label='Rolling 252-day returns',
        tab_style={'margin-left': 'auto', 'margin-right': 'auto'}
    )
]

# Annual returns tab
annRetsTab = [
    dbc.Tab(
        dash_table.DataTable(
            id='dfDashbAnn',
            columns=[
                {
                    'name': i,
                    'id': i,
                    'type': 'numeric',
                    'format': Format(precision=4, scheme=Scheme.fixed)
                }

                for i in tickerClose_retsAnn.columns
            ],
            data=tickerClose_retsAnn.to_dict('records'),
            export_format='csv',
            export_headers='display',

            # Formatting options
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            column_selectable='multi',
            row_selectable='multi',
            page_action='native',
            style_cell={
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '14px',
                'padding': '8px',
                'textAlign': 'center'
            },
            style_header={
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'},
                {'if': {'row_index': 'even'}, 'backgroundColor': 'white'},
                {'if': {'column_id': 'year'}, 'textAlign': 'center', 'fontWeight': 'bold'}
            ],
            style_as_list_view=True
        ),
        label='Annual returns',
        tab_style={'margin-left': 'auto', 'margin-right': 'auto'}
    )
]

# Yearly trades tab
# tradesTab = [
#     dbc.Tab(
#         [
#             # html.H4('Yearly trades', style={'margin-top': '20px'}),
#             dash_table.DataTable(
#                 id='dfDashbTrades',
#                 columns=[
#                     {
#                         'name': i,
#                         'id': i,
#                     }
#
#                     for i in trades_df.columns
#                 ],
#                 data=trades_df.to_dict('records'),
#                 export_format='csv',
#                 export_headers='display',
#
#                 # Formatting options
#                 filter_action='native',
#                 sort_action='native',
#                 sort_mode='multi',
#                 column_selectable='multi',
#                 row_selectable='multi',
#                 page_action='native',
#                 style_cell={
#                     'fontFamily': 'Arial, sans-serif',
#                     'fontSize': '14px',
#                     'padding': '8px',
#                     'textAlign': 'center'
#                 },
#                 style_header={
#                     'fontWeight': 'bold',
#                     'textAlign': 'center'
#                 },
#                 style_data_conditional=[
#                     {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'},
#                     {'if': {'row_index': 'even'}, 'backgroundColor': 'white'},
#                     {'if': {'column_id': 'year'}, 'textAlign': 'center', 'fontWeight': 'bold'}
#                 ],
#                 style_as_list_view=True
#             )
#         ],
#         label='Yearly trades',
#         tab_style={'margin-left': 'auto', 'margin-right': 'auto'}
#     )
# ]

# Performance statistics tab
dfDashb252d = dbc.Col(
    dbc.Card(
        dbc.CardBody(
            [
                html.H5('Results based on 252-day rolling returns', className='dfDashb252d'),
                dash_table.DataTable(
                    id='resdfDashb252d',
                    columns=[
                        {
                            'name': i,
                            'id': i,
                            'type': 'numeric',
                            'format': Format(precision=4, scheme=Scheme.fixed)
                        }

                        for i in performStat252d.columns
                    ],
                    data=performStat252d.to_dict('records'),
                    export_format='csv',
                    export_headers='display',

                    # Formatting options
                    style_cell={
                        'fontFamily': 'Arial, sans-serif',
                        'fontSize': '14px',
                        'padding': '8px',
                        'textAlign': 'center'
                    },
                    style_header={
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'},
                        {'if': {'row_index': 'even'}, 'backgroundColor': 'white'},
                    ],
                    style_as_list_view=True
                )
            ]
        )
    )
)

dfDashbAnn = dbc.Col(
    dbc.Card(
        dbc.CardBody(
            [
                html.H5('Results based on annual returns', className='dfDashbAnn'),
                dash_table.DataTable(
                    id='resdfDashbAnn',
                    columns=[
                        {
                            'name': i,
                            'id': i,
                            'type': 'numeric',
                            'format': Format(precision=4, scheme=Scheme.fixed)
                        }

                        for i in performStatAnn.columns
                    ],
                    data=performStatAnn.to_dict('records'),
                    export_format='csv',
                    export_headers='display',

                    # Formatting options
                    style_cell={
                        'fontFamily': 'Arial, sans-serif',
                        'fontSize': '14px',
                        'padding': '8px',
                        'textAlign': 'center'
                    },
                    style_header={
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'},
                        {'if': {'row_index': 'even'}, 'backgroundColor': 'white'},
                    ],
                    style_as_list_view=True
                )
            ]
        )
    )
)

dfDashbCagr = dbc.Col(
    dbc.Card(
        dbc.CardBody(
            [
                html.H5('Results based on CAGR and annualised standard deviation of daily returns', className='dfDashbCagr'),
                dash_table.DataTable(
                    id='resdfDashbCagr',
                    columns=[
                        {
                            'name': i,
                            'id': i,
                            'type': 'numeric',
                            'format': Format(precision=4, scheme=Scheme.fixed)
                        }

                        for i in performStatCagr.columns
                    ],
                    data=performStatCagr.to_dict('records'),
                    export_format='csv',
                    export_headers='display',

                    # Formatting options
                    style_cell={
                        'fontFamily': 'Arial, sans-serif',
                        'fontSize': '14px',
                        'padding': '8px',
                        'textAlign': 'center'
                    },
                    style_header={
                        'fontWeight': 'bold',
                        'textAlign': 'center'
                    },
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'},
                        {'if': {'row_index': 'even'}, 'backgroundColor': 'white'},
                    ],
                    style_as_list_view=True
                )
            ]
        )
    )
)

performStatTab = [
    dbc.Tab(
        [
            dbc.Row(
                dfDashb252d, style={'margin-bottom': '20px'}
            ),
            dbc.Row(
                dfDashbAnn, style={'margin-bottom': '20px'}
            ),
            dbc.Row(
                dfDashbCagr, style={'margin-bottom': '20px'}
            ),
        ],
        label='Performance statistics',
        tab_style={'margin-left': 'auto', 'margin-right': 'auto'}
    )
]

# App layout
app.layout = dbc.Container(
    [
        # Dashboard title
        dbc.Row(
            html.H1(
                'Long Moving Average Short Short Backtest',
                style={'font-size': '32px',
                       'margin-bottom': '20px',
                       'text-align': 'center'
                }
            )
        ),
        dbc.Row(
            [
                # Inputs section
                dbc.Col(ticker + startDate + endDate + movingAvg,
                        style={'position': 'sticky',
                               'top': '70px',
                               'background-color': 'white',
                               'padding': '10px',
                               'height': 'calc(100vh - 120px)',
                               'overflow-y': 'auto',
                               'overflow-x': 'hidden',
                               'z-index': 1,
                               'border': '1px solid #ccc',
                               'border-radius': '20px'
                        },
                        md=2
                ),
                # Dashboard tabs
                dbc.Col(
                    dbc.Tabs(
                        chartsTab + resultsdfTab + roll252dRetsTab + annRetsTab + performStatTab, #+ tradesTab
                        id='tabs',
                        active_tab='my_graph',
                        # style={'height': '100%',
                        #        'overflow-y': 'auto'
                        # }
                    ),
                    md=10,
                )
            ],
            style={'height': 'calc(100vh - 50px)',
                   'overflow-y': 'auto',
                   # 'overflow-x': 'auto'
            }
        ),
    ],
    fluid=True
)

# Define the callback function to update the plot
@app.callback(
    Output('my_graph', 'figure'),
    Output('resultsdf', 'data'),
    Output('resdfDashb252d', 'data'),
    Output('resdfDashbAnn', 'data'),
    Output('resdfDashbCagr', 'data'),
    # Output('dfDashbTrades', 'data'),
    Output('dfDashb252d', 'data'),
    Output('dfDashbAnn', 'data'),
    Input('ticker', 'value'),
    Input('start_date', 'value'),
    Input('end_date', 'value'),
    Input('sma', 'value')
)


def update_plot(ticker_name, start_date, end_date, sma):
    # Update the backtest
    (
        dataCleanUpdated,
        tickerClose_tradingIndicUpdated,
        tickerClose_exposureTradesUpdated,
        tickerClose_retsDaily_filt,
        rets252dUpdated,
        retsAnnUpdated,
        # tradesUpdated
    ) = btfunc.long_ma_short_short_back_test(
        add_ticker=ticker_name,
        price_for_analysis='close',
        bt_start=start_date,
        bt_end=end_date,
        sma_number=sma,
        starting_capital=initStartingCapital
    )

    # Update the Plotly plot
    figUpdated = btfunc.visualise_strategy_returns_plotly(df=tickerClose_retsDaily_filt, ticker_Name=ticker_name)

    # Update performance statistics
    (
        performStat252dUpdated,
        performStatAnnUpdated,
        performStatCagrUpdated,
        # trades_dfUpdated
    ) = btfunc.results(
        rets_daily=tickerClose_retsDaily_filt,
        rets_252d=rets252dUpdated,
        rets_annual=retsAnnUpdated,
        # trades_df=tradesUpdated
    )

    return figUpdated, tickerClose_retsDaily_filt.to_dict('records'), performStat252dUpdated.to_dict('records'), performStatAnnUpdated.to_dict('records'), performStatCagrUpdated.to_dict('records'), rets252dUpdated.to_dict('records'), retsAnnUpdated.to_dict('records') # , trades_dfUpdated.to_dict('records')

# Run the app
if __name__ == '__main__':
    app.run(debug=False, port=8050, use_reloader=False)
    