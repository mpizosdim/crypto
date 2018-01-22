import scipy.optimize as spo
from project_1a import get_data, compute_daily_returns, normalize_data, assess_portfolio
import numpy as np


def compute_sharpe_ratio(allocs, normed, frequency, annual_daily_rf_bank):
    alloced = allocs * normed
    port_val = alloced.sum(axis=1)

    port_rets = compute_daily_returns(port_val)[1:]
    std_daily_ret = port_rets.std()
    K = frequency ** (1.0 / 2)
    daily_rf = ((1 + annual_daily_rf_bank) ** (1.0 / frequency)) - 1
    sharp_ratio = K * ((port_rets - daily_rf).mean() / std_daily_ret)
    return sharp_ratio * (-1)


def optimize_portfolio(start_date, end_date, symbols, frequency, starting_value, annual_daily_rf_bank, plot=False):
    df = get_data(symbols, start_date, end_date)
    normed = normalize_data(df)

    # initial guess
    guess_allocs = [(1. / len(symbols))] * len(symbols)
    bnds = ((0., 1.),) * len(symbols)

    allocs = spo.minimize(compute_sharpe_ratio, guess_allocs,
                          args=(normed, frequency, annual_daily_rf_bank,),
                          method='SLSQP', options={'disp':True}, bounds=bnds,
                          constraints=({'type': 'eq', 'fun': lambda allocs: 1.0 - np.sum(allocs)}))

    cr, apr, sdr, sr, evp = assess_portfolio(start_date, end_date, symbols, allocs.x, starting_value, annual_daily_rf_bank, frequency, plot)
    return cr, apr, sdr, sr, evp, allocs.x


if __name__ == '__main__':
    start_date = '2010-01-01'
    end_date = '2010-12-31'
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    sv = 1000000
    sf = 252.0
    rfr = 0.0
    cr, apr, sdr, sr, evp, allocs = optimize_portfolio(start_date, end_date, symbols, sf, sv, rfr, True)
    print("cummulative return: %s" % cr)
    print("avg period return: %s" % apr)
    print("std daily return: %s" % sdr)
    print("sharpe ratio: %s" % sr)
    print("current portofolio value: %s" % evp)
    print("allocations: %s" %allocs)