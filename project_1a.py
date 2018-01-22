import pandas as pd
import matplotlib.pyplot as plt

_FILE_PATHS = './data/'


def symbol_to_path(symbol):
    """Return CSV file path given ticker symbol."""
    return "data/{}.csv".format(str(symbol))


def get_data(symbols, start_date, end_date):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)
    for symbol in symbols:  # uses data in data directory, if you want to see a stock not there add it
        df_sym = pd.read_csv(symbol_to_path(symbol), index_col="Date",
                             parse_dates=True, usecols=['Date', 'Adj Close'],
                             na_values=['nan'])
        df_sym = df_sym.rename(columns={'Adj Close': symbol})
        df = df.join(df_sym)
    df = df.dropna()
    return df


def normalize_data(df):
    """normalizes the data"""
    return df / df.ix[0, :]


def compute_daily_returns(df):
    """
    compute and return the daily return values 
    """
    daily_return = (df / df.shift(1)) - 1
    if daily_return.shape.__len__() == 1:
        daily_return.ix[0] = 0
    else:
        daily_return.ix[0, :] = 0
    return daily_return


def compute_daily_portofolio_values(start_date, end_date, symbols, allocs, initial_capital):
    df = get_data(symbols, start_date, end_date)
    norm = normalize_data(df)
    alloced = norm * allocs
    pos_vals = alloced * initial_capital
    port_val = pos_vals.sum(axis=1)
    return port_val


def compute_portfolio_statistics(port_val, frequency, annual_daily_rf_bank=0.1):
    port_rets = compute_daily_returns(port_val)[1:]
    cum_ret = (port_val[-1] / port_val[0] - 1)
    avg_daily_ret = port_rets.mean()
    std_daily_ret = port_rets.std()
    K = frequency ** (1.0 / 2)
    daily_rf = ((1 + annual_daily_rf_bank) ** (1.0 / frequency)) - 1
    sharp_ratio = K * ((port_rets - daily_rf).mean() / std_daily_ret)
    return cum_ret, avg_daily_ret, std_daily_ret, sharp_ratio


def assess_portfolio(start_date, end_date, symbols, allocs, start_value_portofolio, risk_free_return, sampling_freq, plot=False):
    """
    :param start_date: A datetime object that represents the start date 
    :param end_date: A datetime object that represents the end date
    :param symbols:  A list of 2 or more symbols that make up the portfolio (note that your code should support any symbol in the data directory)
    :param allocs: A list of 2 or more allocations to the stocks, must sum to 1.0 
    :param start_value_portofolio: Start value of the portfolio
    :param risk_free_return: The risk free return per sample period that does not change for the entire date range (a single number, not an array).
    :param sampling_freq: Sampling frequency per year
    :param gen_plot: If False, do not create any output. If True it is OK to output a plot such as plot.png 
    :return: 
     cum_ret: Cumulative return
     avg_period_return: Average period return (if sf == 252 this is daily return)
     std_daily_return: Standard deviation of daily return
     sharpe_ratio: Sharpe ratio
     end_value_portofolio: End value of portfolio
    """
    port_val = compute_daily_portofolio_values(start_date, end_date, symbols, allocs, start_value_portofolio)
    cum_ret, avg_period_return, std_daily_return, sharpe_ratio = compute_portfolio_statistics(port_val, sampling_freq, risk_free_return)
    end_value_portofolio = (1 + cum_ret) * start_value_portofolio
    if plot:
        ax = normalize_data(port_val).plot(title="Daily Portfolio Value vs. S&P 500", label='Portfolio')
        spy = get_data(['SPY'], start_date, end_date)
        normed_SPY = normalize_data(spy)
        normed_SPY.plot(label="SPY", ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc='upper left')
        plt.show()
    return cum_ret, avg_period_return, std_daily_return, sharpe_ratio,  end_value_portofolio

if __name__ == '__main__':
    start_date = '2010-01-01'
    end_date = '2010-12-31'
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    Allocations = [0.2, 0.3, 0.4, 0.1]
    sv = 1000000
    sf = 252.0
    rfr = 0.0
    cr, apr, sdr, sr, evp = assess_portfolio(start_date, end_date, symbols, Allocations, sv, rfr, sf, True)
    print("cummulative return: %s" %cr)
    print("avg period return: %s" % apr)
    print("std daily return: %s" % sdr)
    print("sharpe ratio: %s" % sr)
    print("current portofolio value: %s" % evp)


