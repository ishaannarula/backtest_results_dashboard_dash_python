import pandas as pd
import numpy as np
import yfinance as yf
import holoviews as hv
import warnings
import plotly.graph_objects as go
hv.extension('bokeh')

from bokeh.models import HoverTool
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')



def import_and_clean_data(ticker, price_types):
    series = yf.download(ticker, interval='1d')[price_types]

    df = pd.DataFrame({'open': series['Open'],
                       'close': series['Close']})  # modify code here to accept only open or only close prices as well

    print('Total no. of rows in the original dataframe:', len(df))
    print('')

    # Find and delete missing values
    missingvals = df[df.isna().any(axis=1)]

    print('No. of rows with missing values in the original dataframe:', len(missingvals))
    print('')

    print('Missing values in the dataframe:')
    print(missingvals)
    print('')

    print('Deleting rows with missing values...')
    print('')

    print('Preparing clean dataset...')
    print('')

    dfclean = df.drop(index=missingvals.index, errors='ignore')

    print('No. of rows in cleaned dataframe:', len(dfclean))
    print('')

    # print('Clean dataframe:')

    return dfclean


def trading_indicators(df_prev, price_type, moving_avg):
    df = df_prev.copy()
    df[price_type + '_sma'] = df[price_type].rolling(window=moving_avg).mean()

    df[price_type + '_ema12'] = df[price_type].ewm(alpha=2 / (12 + 1), min_periods=12, adjust=False).mean()
    df[price_type + '_ema26'] = df[price_type].ewm(alpha=2 / (26 + 1), min_periods=26, adjust=False).mean()

    df['macd_12,26'] = df[price_type + '_ema12'] - df[price_type + '_ema26']
    df['macd_ema9'] = df['macd_12,26'].ewm(alpha=2 / (9 + 1), min_periods=9, adjust=False).mean()

    df = df.dropna()

    return df


def strategy_exposure_and_trades(df_prev, price_type, moving_avg):
    df = df_prev[['open', 'close', price_type + '_sma', 'macd_12,26', 'macd_ema9']]

    def long_short_signal(row):
        signal = None

        if row['macd_12,26'] > row['macd_ema9']:
            signal = 1
        elif row['macd_12,26'] < row['macd_ema9']:
            signal = -1
        else:
            signal = 0

        return signal

    df['l-s_signal'] = df.apply(lambda row: long_short_signal(row), axis=1)

    print('Data points which produce neither a long nor a short signal:')

    ls_signal_zero = df[df['l-s_signal'] == 0]
    print(ls_signal_zero.to_string())

    def exposure(row):
        exposure = None

        if (
                row[price_type] > row[price_type + '_sma'] and
                row['l-s_signal'] == 1
        ):
            exposure = row['l-s_signal']

        elif (
                row[price_type] < row[price_type + '_sma'] and
                row['l-s_signal'] == -1
        ):
            exposure = row['l-s_signal']

        else:
            exposure = 0

        return exposure

    df['exposure'] = df.apply(lambda row: exposure(row), axis=1)

    # Analysis based on close => exposure to index on day i is the exposure value calculated on day (i-1)
    if price_type == 'close':
        df['exposure'] = df['exposure'].shift(1)

    df['trade'] = df['exposure'] - df['exposure'].shift(1)

    # To make sure that we execute the trade on day i, following which the exposure gets revised on day (i+1)
    df['trade'] = df['trade'].shift(-1)

    df.dropna(inplace=True)

    return df


def strategy_returns(df_prev, returns_based_on, start, end):
    df = df_prev[(df_prev.index >= start) & (df_prev.index <= end)]

    if returns_based_on == 'close':
        df['index_dailyLogRet'] = np.log(df['close'] / df['close'].shift(1))  # ln(close on day i/ close on day i-1)
        df['strat_dailyLogRet'] = df['exposure'] * df['index_dailyLogRet']

        df['strat_cumulRet'] = df['strat_dailyLogRet'].cumsum()
        df['B&H_cumulRet'] = df['index_dailyLogRet'].cumsum()

        # df['strat_portf'] = np.nan
        df['strat_portf'] = 1 * np.exp(df['strat_cumulRet'])

        # if 'strat_portf' in df:
        df.iloc[0, df.columns.get_loc('strat_portf')] = 1

        # df['B&H_portf'] = np.nan
        df['B&H_portf'] = 1 * np.exp(df['B&H_cumulRet'])
        df['B&H_portf'].iloc[0] = 1

    elif returns_based_on == 'open':
        df['index_dailyLogRet'] = np.log(df['open'].shift(-1) / df['open'])  # ln(open on day i+1/ open on day i)
        df['strat_dailyLogRet'] = df['exposure'] * df['index_dailyLogRet']

        df['strat_cumulRet'] = df['strat_dailyLogRet'].cumsum()
        df['B&H_cumulRet'] = df['index_dailyLogRet'].cumsum()

        df['strat_portf'] = np.nan
        df['strat_portf'] = 1 * np.exp(df['strat_cumulRet'])
        df['strat_portf'].iloc[0] = 1
        df['strat_portf'].iloc[-1] = df['strat_portf'].iloc[-2]

        df['B&H_portf'] = np.nan
        df['B&H_portf'] = 1 * np.exp(df['B&H_cumulRet'])
        df['B&H_portf'].iloc[0] = 1
        df['B&H_portf'].iloc[-1] = df['B&H_portf'].iloc[-2]

    # elif returns_based_on == 'both':
    #     df['index_dailyLogRet'] = np.log(df['close']/ df['open']) # ln(close on day i/ open on day i)

    # Add a new column to display row indices (dates)
    df.insert(0, 'date', df.index.date)

    return df


def strategy_returns_rolling_252_days(df):
    dfNew = pd.DataFrame()
    dfNew['strat_252dRet'] = np.exp(df['strat_dailyLogRet'].rolling(window=252).sum()) - 1
    dfNew['B&H_252dRet'] = np.exp(df['index_dailyLogRet'].rolling(window=252).sum()) - 1
    dfNew.dropna(inplace=True)

    # Add a new column to display row indices (dates)
    dfNew.insert(0, 'date', dfNew.index.date)

    return dfNew


def strategy_returns_annual(df):
    dfNew = pd.DataFrame()
    dfNew['strat_annRet'] = np.exp(df['strat_dailyLogRet'].groupby(pd.Grouper(freq='Y')).sum()) - 1
    dfNew['B&H_annRet'] = np.exp(df['index_dailyLogRet'].groupby(pd.Grouper(freq='Y')).sum()) - 1

    # Add a new column to display row indices (dates)
    dfNew.insert(0, 'year', dfNew.index.strftime('%Y'))

    return dfNew


def compunded_annual_growth_rate(df):
    num_years = len(df) / 252  # i.e. Total number of trading days/ trading days in 1 year

    stratFinalVal = df['strat_portf'].iloc[-1]
    stratInitVal = df['strat_portf'].iloc[0]

    stratTotalRet = stratFinalVal / stratInitVal - 1

    BHFinalVal = df['B&H_portf'].iloc[-1]
    BHInitVal = df['B&H_portf'].iloc[0]

    BHTotalRet = BHFinalVal / BHInitVal - 1

    stratCagr = (1 + stratTotalRet) ** (1 / num_years) - 1
    BHCagr = (1 + BHTotalRet) ** (1 / num_years) - 1

    return stratTotalRet, BHTotalRet, stratCagr, BHCagr


def number_of_trades(df):
    dfNew = pd.DataFrame()

    # Calculate annual no. of trades executed
    trade_yesNo_daily = (df['trade'].abs() > 0)
    trade_yesNo_ann = trade_yesNo_daily.groupby(pd.Grouper(freq='Y')).sum()

    dfNew['numTrades'] = trade_yesNo_ann

    # Calculate annual no. longs and shorts
    long_yesNo_daily = (df['trade'] > 0)
    long_yesNo_ann = long_yesNo_daily.groupby(pd.Grouper(freq='Y')).sum()

    dfNew['numLongs'] = long_yesNo_ann

    # Calculate annual no. shorts
    short_yesNo_daily = (df['trade'] < 0)
    short_yesNo_ann = short_yesNo_daily.groupby(pd.Grouper(freq='Y')).sum()

    dfNew['numShorts'] = short_yesNo_ann

    # Calculate annual long/ short exposure to index
    dfNew['l-s_exposure'] = dfNew['numLongs'] - dfNew['numShorts']

    # Add a new column to display row indices (dates)
    dfNew.insert(0, 'year', dfNew.index.strftime('%Y'))

    return dfNew


def long_long_short_short_back_test(add_ticker, price_for_analysis, bt_start, bt_end, sma_number):
    # Data cleaning
    print('DATA CLEANING:')
    dataClean = import_and_clean_data(ticker=add_ticker, price_types=['Open', 'Close'])
    print('')

    # Trading indicators
    tradingIndic = trading_indicators(df_prev=dataClean, price_type=price_for_analysis, moving_avg=sma_number)

    # Exposure and trades
    print('EXPOSURE AND TRADES:')
    exposureTrades = strategy_exposure_and_trades(df_prev=tradingIndic, price_type=price_for_analysis,
                                                  moving_avg=sma_number)
    print('')

    # Returns - daily
    retsDaily = strategy_returns(df_prev=exposureTrades, returns_based_on=price_for_analysis, start=bt_start,
                                 end=bt_end)

    # Returns - rolling 252-day, annual
    rets252d = strategy_returns_rolling_252_days(retsDaily)
    retsAnn = strategy_returns_annual(retsDaily)

    # Trades
    trades = number_of_trades(retsDaily)

    return (dataClean, tradingIndic, exposureTrades, retsDaily, rets252d, retsAnn, trades)


def visualise_strategy_returns(df, price_type, moving_avg):
    figure_size = {'width': 1200, 'height': 500}

    hover = HoverTool(tooltips=[('Date', '@date{%F}'),
                                ('Value', '$y{0.4F}')],
                      formatters={'@date': 'datetime'},
                      mode='vline')

    df['date'] = df.index

    index = hv.Curve(df, 'date', price_type, label='Index ' + price_type).opts(tools=[hover])
    index_sma = hv.Curve(df, 'date', price_type + '_sma',
                         label='Index ' + price_type + ' ' + str(moving_avg) + '-day MA').opts(tools=[hover])

    overlay1 = index * index_sma
    overlay1 = overlay1.opts(title='Price and its 200-day Moving Average',
                             xlabel='Date',
                             ylabel='Prices',
                             legend_position='right',
                             show_grid=True,
                             **figure_size)

    macd_1226 = hv.Curve(df, 'date', 'macd_12,26', label='MACD = 12-day EMA - 26-day EMA').opts(tools=[hover])
    macd_ema9 = hv.Curve(df, 'date', 'macd_ema9', label='9-day EMA of MACD').opts(tools=[hover])

    overlay2 = macd_1226 * macd_ema9
    overlay2 = overlay2.opts(
        title='MACD (Difference between 12-day EMA and 26-day EMA of the Index) and 9-day EMA of MACD',
        xlabel='Date',
        ylabel='Values',
        legend_position='right',
        show_grid=True,
        **figure_size)

    overlay3 = hv.Curve(df, 'date', 'exposure', label='Exposure').opts(tools=[hover])
    overlay3 = overlay3.opts(title='Strategy Exposure',
                             xlabel='Date',
                             ylabel='Exposure',
                             show_grid=True,
                             **figure_size)

    index_dailyLogRet = hv.Curve(df, 'date', 'index_dailyLogRet', label='Buy-and-Hold Daily Log Return').opts(
        color='lightgrey', tools=[hover])
    strat_dailyLogRet = hv.Curve(df, 'date', 'strat_dailyLogRet', label='Strategy Daily Log Return').opts(
        color='#1f77b4', tools=[hover])

    overlay4 = index_dailyLogRet * strat_dailyLogRet
    overlay4 = overlay4.opts(title='Daily Strategy and Buy-and-Hold Returns',
                             xlabel='Date',
                             ylabel='Log Returns',
                             legend_position='right',
                             show_grid=True,
                             **figure_size)

    strat_portf = hv.Curve(df, 'date', 'strat_portf', label='Strategy Portfolio').opts(tools=[hover])
    BH_portf = hv.Curve(df, 'date', 'B&H_portf', label='Buy-and-Hold Portfolio').opts(tools=[hover])

    overlay5 = strat_portf * BH_portf
    overlay5 = overlay5.opts(title='Evolution of Strategy and Buy-and-Hold Portfolios',
                             xlabel='Date',
                             ylabel='$1 Portfolio Value',
                             legend_position='right',
                             show_grid=True,
                             **figure_size)

    subplots = hv.Layout(overlay1 + overlay2 + overlay3 + overlay4 + overlay5).cols(1)
    df.drop('date', axis=1)

    return subplots


def visualise_strategy_returns_plotly(df, moving_avg, ticker_Name):
    # Create subplots
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=(
            'Closing price and its moving average',
            'MACD and the 9-day EMA of the MACD',
            'Strategy exposure',
            'Daily log returns',
            'Portfolio evolution'
        ),
        vertical_spacing=0.02,
        shared_xaxes=True
    )

    # Add traces to each subplot
    # Subplot 1
    fig.add_trace(
        go.Scatter(x=df.index, y=df['close'], name=str(ticker_Name) + ' close', legendgroup='1'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['close_sma'], name='Moving average', legendgroup='1'),
        row=1, col=1
    )

    # Subplot 2
    fig.add_trace(
        go.Scatter(x=df.index, y=df['macd_12,26'], name='MACD = 12-day EMA - 26-day EMA', legendgroup='2'),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['macd_ema9'], name='9-day EMA of MACD', legendgroup='2'),
        row=2, col=1
    )

    # Subplot 3
    fig.add_trace(
        go.Scatter(x=df.index, y=df['exposure'], name='Exposure', legendgroup='3'),
        row=3, col=1
    )

    # Subplot 4
    fig.add_trace(
        go.Scatter(x=df.index, y=df['index_dailyLogRet'], name='Index daily log return', legendgroup='4'),
        row=4, col=1
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['strat_dailyLogRet'], name='Strategy daily log return', legendgroup='4'),
        row=4, col=1
    )

    # Subplot 5
    fig.add_trace(
        go.Scatter(x=df.index, y=df['B&H_portf'], name='Buy and hold portfolio', legendgroup='5'),
        row=5, col=1
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df['strat_portf'], name='Strategy portfolio', legendgroup='5'),
        row=5, col=1
    )

    # Add subplot titles
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Trading indicators', row=2, col=1)
    fig.update_yaxes(title_text='Position', row=3, col=1)
    fig.update_yaxes(title_text='Log return', row=4, col=1)
    fig.update_yaxes(title_text='Capital', row=5, col=1)

    # Update layout
    fig.update_layout(
        height=3000, width=1350, legend_tracegroupgap=600,
        xaxis_showticklabels=True,
        xaxis2_showticklabels=True,
        xaxis3_showticklabels=True,
        xaxis4_showticklabels=True
    )

    return fig


def results(rets_daily, rets_252d, rets_annual, trades_df):
    # Average return, standard deviation, Sharpe and Sortino ratios based on rolling 252-day returns

    avg252dStratRet = rets_252d['strat_252dRet'].mean()
    avg252dBHRet = rets_252d['B&H_252dRet'].mean()

    std252dStratRet = rets_252d['strat_252dRet'].std()
    std252dBHRet = rets_252d['B&H_252dRet'].std()

    stdDown252dStratRet = rets_252d['strat_252dRet'][rets_252d['strat_252dRet'] < 0].std()
    stdDown252dBHRet = rets_252d['B&H_252dRet'][rets_252d['B&H_252dRet'] < 0].std()

    sr252dStratRet = avg252dStratRet / std252dStratRet
    sr252dBHRet = avg252dBHRet / std252dBHRet

    sortino252dStratRet = avg252dStratRet / stdDown252dStratRet
    sortino252dBHRet = avg252dBHRet / stdDown252dBHRet

    # Dataframe
    performStat252d = pd.DataFrame(columns=['Strategy', 'Buy & hold'],
                                   index=['Avg. Return', 'Vol.', 'Vol. -ve Return', 'Sharpe', 'Sortino'])

    performStat252d.loc['Avg. Return', 'Strategy'] = avg252dStratRet
    performStat252d.loc['Avg. Return', 'Buy & hold'] = avg252dBHRet

    performStat252d.loc['Vol.', 'Strategy'] = std252dStratRet
    performStat252d.loc['Vol.', 'Buy & hold'] = std252dBHRet

    performStat252d.loc['Vol. -ve Return', 'Strategy'] = stdDown252dStratRet
    performStat252d.loc['Vol. -ve Return', 'Buy & hold'] = stdDown252dBHRet

    performStat252d.loc['Sharpe', 'Strategy'] = sr252dStratRet
    performStat252d.loc['Sharpe', 'Buy & hold'] = sr252dBHRet

    performStat252d.loc['Sortino', 'Strategy'] = sortino252dStratRet
    performStat252d.loc['Sortino', 'Buy & hold'] = sortino252dBHRet

    # Formatting
    performStat252d.loc['Avg. Return', :] = performStat252d.loc['Avg. Return', :].apply(lambda x: '{:.2%}'.format(x))
    performStat252d.loc['Vol.', :] = performStat252d.loc['Vol.', :].apply(lambda x: '{:.2%}'.format(x))
    performStat252d.loc['Vol. -ve Return', :] = performStat252d.loc['Vol. -ve Return', :].apply(
        lambda x: '{:.2%}'.format(x))

    # Add a new column to display row indices (metrics)
    performStat252d.insert(0, 'Metrics', performStat252d.index)

    # Average return, standard deviation, Sharpe and Sortino ratios based on annual returns

    avgAnnStratRet = rets_annual['strat_annRet'].mean()
    avgAnnBHRet = rets_annual['B&H_annRet'].mean()

    stdAnnStratRet = rets_annual['strat_annRet'].std()
    stdAnnBHRet = rets_annual['B&H_annRet'].std()

    stdDownAnnStratRet = rets_annual['strat_annRet'][rets_annual['strat_annRet'] < 0].std()
    stdDownAnnBHRet = rets_annual['B&H_annRet'][rets_annual['B&H_annRet'] < 0].std()

    srAnnStratRet = avgAnnStratRet / stdAnnStratRet
    srAnnBHRet = avgAnnBHRet / stdAnnBHRet

    sortinoAnnStratRet = avgAnnStratRet / stdDownAnnStratRet
    sortinoAnnBHRet = avgAnnBHRet / stdDownAnnBHRet

    # Dataframe
    performStatAnn = pd.DataFrame(columns=['Strategy', 'Buy & hold'],
                                  index=['Avg. Return', 'Vol.', 'Vol. -ve Return', 'Sharpe', 'Sortino'])

    performStatAnn.loc['Avg. Return', 'Strategy'] = avgAnnStratRet
    performStatAnn.loc['Avg. Return', 'Buy & hold'] = avgAnnBHRet

    performStatAnn.loc['Vol.', 'Strategy'] = stdAnnStratRet
    performStatAnn.loc['Vol.', 'Buy & hold'] = stdAnnBHRet

    performStatAnn.loc['Vol. -ve Return', 'Strategy'] = stdDownAnnStratRet
    performStatAnn.loc['Vol. -ve Return', 'Buy & hold'] = stdDownAnnBHRet

    performStatAnn.loc['Sharpe', 'Strategy'] = srAnnStratRet
    performStatAnn.loc['Sharpe', 'Buy & hold'] = srAnnBHRet

    performStatAnn.loc['Sortino', 'Strategy'] = sortinoAnnStratRet
    performStatAnn.loc['Sortino', 'Buy & hold'] = sortinoAnnBHRet

    # Formatting
    performStatAnn.loc['Avg. Return', :] = performStatAnn.loc['Avg. Return', :].apply(lambda x: '{:.2%}'.format(x))
    performStatAnn.loc['Vol.', :] = performStatAnn.loc['Vol.', :].apply(lambda x: '{:.2%}'.format(x))
    performStatAnn.loc['Vol. -ve Return', :] = performStatAnn.loc['Vol. -ve Return', :].apply(
        lambda x: '{:.2%}'.format(x))

    # Add a new column to display row indices (metrics)
    performStatAnn.insert(0, 'Metrics', performStatAnn.index)

    # Average return based on CAGR, annualised standard deviation based on daily returns, Sharpe and Sortino ratios

    totalretStrat, totalRetBH, cagrStrat, cagrBH = compunded_annual_growth_rate(df=rets_daily)

    stdDailyStratRet = rets_daily['strat_dailyLogRet'].std() * np.sqrt(252)  # annualised measure
    stdDailyBHRet = rets_daily['index_dailyLogRet'].std() * np.sqrt(252)  # annualised measure

    stdDownDailyStratRet = rets_daily['strat_dailyLogRet'][rets_daily['strat_dailyLogRet'] < 0].std() * np.sqrt(
        252)  # annualised measure
    stdDownDailyBHRet = rets_daily['index_dailyLogRet'][rets_daily['index_dailyLogRet'] < 0].std() * np.sqrt(
        252)  # annualised measure

    srStratCagrStd = cagrStrat / stdDailyStratRet
    srBHCagrStd = cagrBH / stdDailyBHRet

    sortinoStratCagrStd = cagrStrat / stdDownDailyStratRet
    sortinoBHCagrStd = cagrBH / stdDownDailyBHRet

    # Dataframe
    performStatCagr = pd.DataFrame(columns=['Strategy', 'Buy & hold'],
                                   index=['CAGR', 'Vol.', 'Vol. -ve Return', 'Sharpe', 'Sortino'])

    performStatCagr.loc['CAGR', 'Strategy'] = cagrStrat
    performStatCagr.loc['CAGR', 'Buy & hold'] = cagrBH

    performStatCagr.loc['Vol.', 'Strategy'] = stdDailyStratRet
    performStatCagr.loc['Vol.', 'Buy & hold'] = stdDailyBHRet

    performStatCagr.loc['Vol. -ve Return', 'Strategy'] = stdDownDailyStratRet
    performStatCagr.loc['Vol. -ve Return', 'Buy & hold'] = stdDownDailyBHRet

    performStatCagr.loc['Sharpe', 'Strategy'] = srStratCagrStd
    performStatCagr.loc['Sharpe', 'Buy & hold'] = srBHCagrStd

    performStatCagr.loc['Sortino', 'Strategy'] = sortinoStratCagrStd
    performStatCagr.loc['Sortino', 'Buy & hold'] = sortinoBHCagrStd

    # Formatting
    performStatCagr.loc['CAGR', :] = performStatCagr.loc['CAGR', :].apply(lambda x: '{:.2%}'.format(x))
    performStatCagr.loc['Vol.', :] = performStatCagr.loc['Vol.', :].apply(lambda x: '{:.2%}'.format(x))
    performStatCagr.loc['Vol. -ve Return', :] = performStatCagr.loc['Vol. -ve Return', :].apply(
        lambda x: '{:.2%}'.format(x))

    # Add a new column to display row indices (metrics)
    performStatCagr.insert(0, 'Metrics', performStatCagr.index)

    return (performStat252d, performStatAnn, performStatCagr, trades_df)




