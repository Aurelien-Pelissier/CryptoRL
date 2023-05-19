
import os.path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import parser

from datetime import timedelta, datetime
from binance.client import Client
from binance.enums import HistoricalKlinesType

def main():
    binance_symbols = ["BTCUSDT", "ETHUSDT", "BCHUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT", "ZRXUSDT", "LINKUSDT", "EOSUSDT", "ADAUSDT", "XTZUSDT", "TRXUSDT", "BATUSDT", "ETCUSDT", "XLMUSDT", "XMRUSDT"]
    #binance_symbols = ["ETHBTC", "BCHBTC", "BNBBTC", "XRPBTC", "LTCBTC", "ZRXBTC", "LINKBTC", "EOSBTC", "ADABTC", "XTZBTC", "TRXBTC", "BATBTC", "ETCBTC", "XLMBTC", "XMRBTC"]
    #binance_symbols = ["LINKBTC","LENDBTC","KNCBTC"]
    #binance_symbols = ["LINKBTC","LINKUSDT"]
    
    ### API
    binance_api_key = '[REDACTED]'    #Data are accessible without API key
    binance_api_secret = '[REDACTED]' #Data are accessible without API key
    
    ### CONSTANTS
    start_date = None #If None, it will download data from the first available trading
    batch_size = 750
    binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
    kline_size = '1d' #1m,5m,1h,1d
    futures = True
    
    if futures:
        klines_type = HistoricalKlinesType.FUTURES
    else:
        klines_type = HistoricalKlinesType.SPOT
    
    if not os.path.exists(kline_size):
        os.makedirs(kline_size)
    
    if kline_size == '1m':
        start_date = '10/01/2022'
    
    for symbol in binance_symbols:
        get_all_binance(symbol, kline_size, binance_client, klines_type, start_date = start_date, save = True)
        
def minutes_of_new_data(symbol, kline_size, data, binance_client, start_date = None):
    if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1])
    if start_date is None:
        old = datetime.strptime('1 Jan 2017', '%d %b %Y')
    else:
        old = datetime.strptime(start_date, '%m/%d/%Y')
    new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    #client.futures_historical_klines(symbol="XRPBUSD", interval="5m", start_str= "30 minutes ago UTC") 
    return old, new
    
    
def get_all_binance(symbol, kline_size, binance_client, klines_type, start_date = None, save = False):
    binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
    filename = '%s/%s-%s-data.csv' % (kline_size,symbol, kline_size)
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, binance_client, start_date)
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])
    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): print('Downloading all available %s data for %s...' % (kline_size, symbol))
    else: print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data...' % (delta_min, symbol, available_data, kline_size))
    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"), klines_type=klines_type)
    #klines = binance_client.futures_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else: data_df = data
    data_df.set_index('timestamp', inplace=True)
    data_df.drop(columns = ['close_time','quote_av','trades','tb_base_av','tb_quote_av','ignore'], inplace=True)
    if save: data_df.to_csv(filename)
    print('All caught up..!')
    return data_df


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 18})
    main()