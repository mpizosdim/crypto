import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from dateutil import parser as dateutilParser
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

_starting_date = "2010-01-01"
_data_folder = "/home/dimitris//Documents/others/finance_notes/data/crypto/"


def downloadCryptoList100(top=100, with_info=True):
    r = requests.get("https://coinmarketcap.com/", timeout=30)
    soup = BeautifulSoup(r.text, 'html.parser')
    rows = soup.find_all("a", {"class": "currency-name-container"})
    crypto_names = {}
    for row in rows[:top]:
        crypto_name = str(row.contents[0].replace(" ", "-"))
        if with_info:
            r2 = requests.get("https://coinmarketcap.com/currencies/%s"%crypto_name, timeout=30)
            soup2 = BeautifulSoup(r2.text, 'html.parser')
            tags = [x.text for x in soup2.find_all("div", {"class": "container"})[1].find_all("span", {"class": "label label-warning"})]
            crypto_names[crypto_name] = tags
        else:
            crypto_names[crypto_name] = []
    return crypto_names


def symbol_to_path(symbol):
    """Return CSV file path given ticker symbol."""
    return _data_folder +"{}.csv".format(str(symbol))


def load_crypto_data(currencies, col):
    dateNow = datetime.now()
    dates = pd.date_range(_starting_date, dateNow)
    df = pd.DataFrame(index=dates)
    for crp in currencies:  # uses data in data directory, if you want to see a stock not there add it
        df_sym = pd.read_csv(symbol_to_path(crp), index_col="date",
                             parse_dates=True, usecols=["date", col],
                             na_values=['nan'])
        df_sym = df_sym.rename(columns={col: crp})
        df = df.join(df_sym)
    df = df.dropna(how="all")
    return df

def downloadUsdPriceData(currency, fillHoles=False):
    dateNow = datetime.now()
    year = str(dateNow.year)
    month = str(dateNow.month)
    if len(month) < 2:
        month = "0" + month
    day = str(dateNow.day)
    if len(day) < 2:
        day = "0" + day
    endDate = year + month + day
    startDate = "20100101" if currency != "bch" else "20170731"
    r = requests.get("https://coinmarketcap.com/currencies/%s/historical-data/?start=%s&end=%s" % (currency, startDate, endDate), timeout=30)
    soup = BeautifulSoup(r.text, 'html.parser')

    rows = soup.find("div", id="historical-data").find_all("tr", class_="text-right")
    result = []
    for row in rows[::-1]:
        tds = row.find_all("td")

        volume = tds[5].text
        if volume == "-":
            volume = 0.0
        else:
            volume = float(volume.replace(",", ""))

        marketcap = tds[6].text
        if marketcap == "-":
            marketcap = 0.0
        else:
            marketcap = float(marketcap.replace(",", ""))

        date = dateutilParser.parse(tds[0].text)
        price = float(tds[4].text.replace(",", ""))
        result.append((date, price, marketcap, volume))

    if fillHoles:
        while True:
            holesFound = False
            prevDate = result[0][0]
            index = 1
            for row in result[1:]:
                diff = row[0] - prevDate
                if diff > timedelta(days=1):
                    holesFound = True
                    print "Missing price for %s" % (prevDate + timedelta(days=1))
                    result.insert(index, (prevDate + timedelta(days=1), row[1], row[2], row[3]))
                    break
                prevDate = row[0]
                index += 1

            if not holesFound:
                break
    allData = sorted(result, key=lambda elem: elem[0])
    f = open(_data_folder + "%s.csv" % currency, "w")
    f.write("date,price,mrkcap,volume\n")
    for date, price, mrk_cap, volume in allData:
        f.write(date.strftime('%Y-%m-%d') + ",")
        f.write(",".join([str(price), str(mrk_cap), str(volume)]))
        f.write("\n")
    f.close()
    return 1


def dummy_translator(crp):
    crp = crp.lower()
    if crp == "bytecoin":
        crp = "bytecoin-bcn"
    if crp == "basic-attenti...":
        crp = "basic-attention-token"
    if crp == "experience-po...":
        crp = "experience-points"
    if crp == "byteball-bytes":
        crp = "byteball"
    if crp == "nebulas":
        crp = "nebulas-token"
    if crp == "po.et":
        crp = "poet"
    return crp


def normalize_data_by_first(df):
    """normalizes the data"""
    first_days = df.apply(lambda x: x.get_value(x.first_valid_index()))
    return df / first_days


def normalize_data_by_max(df):
    max_values = df.max(axis=0)
    return df / max_values


def compute_percentage_per_day(df):
    perc = (df / df.shift(1)) - 1
    return perc


def replace_by_nan(df, value):
    return df.replace(value, np.nan)


def series_to_supervised(data, column_names, predict_indexes, predict_names, n_in=1, n_out=1, dropnan=True):
    data = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (j, i)) for j in column_names]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data[predict_indexes].shift(-i))
        if i == 0:
            names += [('%s(t)' % (j)) for j in predict_names]
        else:
            names += [('%s(t+%d)' % (j, i)) for j in predict_names]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# =================================================================================================
# =========================================== MAIN SCRIPTS ========================================
# =================================================================================================


def main_crawl_crypto():
    top100crypto = downloadCryptoList100(False)
    for crp in list(top100crypto.keys()):
        crp = dummy_translator(crp)
        count = 1
        success = None
        while success is None:
            print "parsing crypto: %s for time: %s" % (crp, count)
            try:
                success = downloadUsdPriceData(crp)
            except:
                pass
            count += 1


def preprocess_and_create_data():
    top100crypto = downloadCryptoList100(15, False)
    top100crypto_updated = [dummy_translator(crp) for crp in list(top100crypto.keys())]
    top100crypto_updated.sort()
    predict_currencies = ['bitcoin', 'ethereum', 'ripple']
    predict_currencies.sort()
    #prices
    prices_data = load_crypto_data(top100crypto_updated, "price")
    prices_norm_data = normalize_data_by_first(prices_data)
    prices_perc = compute_percentage_per_day(prices_norm_data)

    # pre-processing
    prices_perc = prices_perc.astype("float32")
    column_names = list(prices_perc.columns.values)
    indexes = [i for i, x in enumerate(column_names) if x in predict_currencies]
    prices_perc.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_model = scaler.fit(prices_perc)
    scaled = scaler_model.transform(prices_perc)
    reframed = series_to_supervised(scaled, column_names, indexes, predict_currencies)

    values = reframed.values
    n_train_days = 88#365 * 2
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]
    train_x, train_y = train[:, :-len(indexes)], train[:, len(column_names):]
    test_x, test_y = test[:, :-len(indexes)], test[:, len(column_names):]
    train_X = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_X = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_y, test_X, test_y, scaler_model, indexes



    # #volume
    # volume_data = load_crypto_data(top100crypto_updated, "volume")
    # volume_data = replace_by_nan(volume_data, 0.0)
    # volume_norm_data = normalize_data_by_max(volume_data)
    #
    # #market cap
    # mrkcap_data = load_crypto_data(top100crypto_updated, "mrkcap")
    # mrkcap_data = replace_by_nan(volume_data, 0.0)
    # mrkcap_norm_data = normalize_data_by_max(volume_data)
    #
    #
    # #Top100 values
    # top_100_perc = prices_perc.mean(axis=1)
    # top_100_volume = volume_data.sum(axis=1)
    # top_100_mrk_cap = mrkcap_data.sum(axis=1)

if __name__ == '__main__':
    #main_crawl_crypto()
    data = preprocess_and_create_data()