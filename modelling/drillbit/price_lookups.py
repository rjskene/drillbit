import datetime as dt
import requests
from bs4 import BeautifulSoup as BS

import pandas as pd

from tqdm.auto import tqdm
import xlwings as xw

def lookup_price(name, manufacturer, hr, pluses):
    if manufacturer == 'Bitmain':
        suffix = f'-{hr}th'
    elif manufacturer == 'MicroBT':
        if pluses == 0:
            suffix = ''
        else:
            suffix = f'-{pluses}'
    elif manufacturer == 'Innosilicon':
        suffix = f'-{hr}t'
    else:
        suffix = ''
    
    url = f'https://www.asicminervalue.com/miners/{manufacturer}/{name}{suffix}'
    request = requests.get(url)
    soup = BS(request.text, features='lxml')
    try:
        price_table = soup.find_all('tbody')[3]
        price_txt = price_table.find('tr').find_all('td')[2].find('b')
        if price_txt is None:
            return 'No Pricing'
        else:
            price = price_table.find('tr').find_all('td')[2].find('b').text.replace('$', '').replace(',', '')
            price = float(price)

            return price

    except IndexError as e:
        return 'Not Found'
    except AttributeError as e:
        if "'NoneType' object has no attribute 'find_all'" in str(e):
            return 'Not Found'
        else:
            raise e

@xw.sub
def price_lookup():
    btc = xw.Book.caller()
    ws = btc.sheets['Miners']
    miners = ws.range('MinerDB[[#All]]').options(pd.DataFrame, header=True).value
    prices = pd.Series([], dtype='object')
    pbar = tqdm(miners.iterrows(), total=miners.shape[0])
    for name, miner in pbar:
        pbar.set_description(f'{name}')
        pluses = name.count('+')
        lookup_name = '-'.join(name.split()).lower().replace('+', '')
        hr = int(miner['Hash Rate'])
        manu = miner['Manufacturer']
        price = lookup_price(lookup_name, manu, hr, pluses)
        prices.loc[name] = price

    btc.app.enable_events = False
    ws.range('MinerDB[Price]').value = prices.values.reshape(-1,1)
    ws.api.OLEObjects('PriceButton').Object.Caption = 'Updated: ' + dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    btc.app.enable_events = True

