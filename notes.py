import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

_FILE_PATHS = './data/'


def general_functions_pandas():
    # read csv file
    df = pd.read_csv(_FILE_PATHS + "AAPL.csv")

    # get 5 first rows
    print(df.head(5))

    # get 5 last rows
    print(df.tail(5))

    # slicing: rows between index 10 and 20
    print(df[10:21])

    # plot adj close
    df['Adj Close'].plot()
    plt.show()

    # plot adj close and close
    df[['Adj Close', 'Close']].plot()
    plt.show()

    # calculate rolling mean
    pd.rolling_mean(df['Adj Close'], window=20)


def get_max_close(symbol):
    """
    Return the maximum closing value for stock indicated by symbol
    """
    df = pd.read_csv(_FILE_PATHS+ "{}.csv".format(symbol))
    return df['Close'].max()


def get_mean_volume(symbol):
    """
    Return the mean volume value for stock indicated by symbol
    """
    df = pd.read_csv(_FILE_PATHS + "{}.csv".format(symbol))
    return df['Volume'].mean()


def get_range(symbols, start_date, end_date, column):
    """
    Return specified period of range for list of stocks 
    """
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)
    if "SPY" not in symbols:
        symbols.insert(0, 'SPY')
    for symbol in symbols:
        df_temp = pd.read_csv(_FILE_PATHS + "{}.csv".format(symbol),
                              index_col="Date", parse_dates=True,
                              na_values=['nan'], usecols=['Date', column])
        df_temp = df_temp.rename(columns={column: symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':
            df = df.dropna(subset=['SPY'])
    return df


def get_specific_range(df, start_date, end_date, columns):
    return df.ix[start_date:end_date, columns]


def plot_data(df, title="stock prices"):
    """
    plot stock prices 
    """
    ax = df.plot(title=title, fontsize=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()


def plot_selected(df, start_date, end_date, columns, normalize=True):
    """
    plot selected range 
    """
    df = get_specific_range(df, start_date, end_date, columns)
    if normalize:
        df = normalize_data(df)
    plot_data(df)


def normalize_data(df):
    """
    normalize the data with the first row 
    """
    return df / df.ix[0, :]


def get_rolling_mean(df, window):
    """
    return rolling mean of given values 
    """
    #TODO: maybe change it like this for the future df.rolling(window=10, center=False).mean()
    return pd.rolling_mean(df, window=window)


def get_rolling_std(df, window):
    """
    return rolling std of given values 
        """
    return pd.rolling_std(df, window=window)


def get_bollinger_bands(rm, rstd):
    """
    return upper and lower bollinger bands 
    """
    return rm + 2 * rstd, rm - 2 * rstd


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


def plot_daily_return(df, bins=10, together=False):
    """
    plots histogram of daily returns indipendadly 
    """
    dr = compute_daily_returns(df)
    symbols = list(dr)
    if together:
        for i in range(len(symbols)):
            dr[symbols[i]].hist(bins=bins, label=symbols[i], alpha=0.5)
        plt.legend(loc='upper right')
    else:
        ax = dr.hist(bins=bins)

        for i in range(len(symbols)):
            temp_mean = dr[symbols[i]].mean()
            temp_std = dr[symbols[i]].std()
            temp_kyrtosis = dr[symbols[i]].kurtosis()
            ax[0][i].axvline(temp_mean, color='b', linestyle='dashed', linewidth=2)
            ax[0][i].axvline(temp_mean + temp_std, color='r', linestyle='dashed', linewidth=2)
            ax[0][i].axvline(temp_mean - temp_std, color='r', linestyle='dashed', linewidth=2)
            ax[0][i].text(0, ax[0][i].get_ylim()[1] / 2.0, "mean: %s"%temp_mean)
            ax[0][i].text(0, ax[0][i].get_ylim()[1] / 2.5, "std: %s" % temp_std)
            ax[0][i].text(0, ax[0][i].get_ylim()[1] / 3.0, "kurtosis: %s" % temp_kyrtosis)
    plt.show()


def scatter_plot(df, symbol):
    dr = compute_daily_returns(df)
    correlation = dr.corr(method='pearson')
    ax = dr.plot(kind='scatter', x='SPY', y=symbol)
    beta, alpha = np.polyfit(dr['SPY'], dr[symbol], 1)
    plt.plot(dr['SPY'], beta * dr['SPY'] + alpha, '-', color='r')
    plt.text(0, ax.get_ylim()[1] / 2, "beta: %s"%beta)
    plt.text(0, ax.get_ylim()[1] / 2.5, "alpha: %s" % alpha)
    plt.text(0, ax.get_ylim()[1] / 3.0, "corr: %s" % correlation[symbol][0])
    plt.show()


def fill_na(df):
    """
    fill nan value first forward and after backword 
    """
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df


def compute_daily_portofolio_values(start_date, end_date, symbols, column, allocs, initial_capital):
    df = get_range(symbols, start_date, end_date, column)
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
    if frequency == 'daily':
        K = 252 ** (1.0 / 2)
    elif frequency == 'weekly':
        K = 52 ** (1.0 / 2)
    elif frequency == 'monthly':
        K = 12 ** (1.0 / 2)
    daily_rf = (1 + annual_daily_rf_bank) ** (1.0 / 252) - 1
    sharp_ratio = K * (avg_daily_ret - daily_rf) / std_daily_ret

    return cum_ret, avg_daily_ret, std_daily_ret, sharp_ratio


def minimizer_example():
    import scipy.optimize as spo
    def f(x):
        Y = (x - 1.5)**2 + 0.5
        return Y
    X_guess = 2.0
    min_result = spo.minimize(f, X_guess, method='SLSQP', options={'disp':True})


def minimizer_example_2():
    import scipy.optimize as spo
    def error(line, data):
        err = np.sum((data[:, 1] - line[0]*data[:,0] + line[1])**2)
        return err

    def fit_line(data, error_fun):
        l = np.float32([0, np.mean(data[:, 1])])
        result = spo.minimize(error_fun, l, args=(data,), method='SLSQP', options={'disp':True})
        return result.x

    def error_poly(C, data):
        err = np.sum((data[:,1] - np.polyval(C, data[:, 0]))**2)
        return err

    def fit_poly(data, error_fun, deg=3):
        Cguess = np.poly1d(np.ones(deg+1, dtype=np.float32))
        result = spo.minimize(error_fun, Cguess, args=(data,), method='SLSQP', options={'disp': True})
        return np.poly1d(result.x)


if __name__ == '__main__':

    port_val = compute_daily_portofolio_values("2009-01-01", "2009-12-31", ["SPY", "AAPL"], "Adj Close", [0.5, 0.5], 1000)
    lol = compute_portfolio_statistics(port_val, "daily", 0.1)
    print(port_val.head(2))
    print(lol)

    #plot_daily_return(test, 20, True)
    #plot_selected(test, "2010-01-01", "2010-09-28", ["SPY", "AAPL"])
    #print(test.head(2))
