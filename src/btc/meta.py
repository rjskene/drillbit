import requests
import datetime as dt
import numpy as np
import pandas as pd
import requests

from .funcs import MiningMixin, decompress

def blockchain_api(chartName, timespan='5years', fmt='csv'):
    # $chartName?timespan=$timespan&rollingAverage=$rollingAverage&start=$start&format=$format&sampled=$sampled'
    root = f'https://api.blockchain.info/charts/{chartName}?timespan={timespan}&format={fmt}'
    return root

def hist_diff():
    hist_diff = 'difficulty.csv'
    df_diff = pd.read_csv(hist_diff).set_index('Timestamp').squeeze()
    df_diff.index = pd.to_datetime(df_diff.index).to_period(freq='D')

    return df_diff

def current_btc_price():
    return pd.read_json('https://blockchain.info/ticker')

def init_meta():
    meta = BTCMeta()
    meta.update_meta()
    return meta

class BTCMeta(MiningMixin):
    INIT_BLOCK_REWARD = 50
    RETARGET_BLOCKS = 2016
    BLOCKS_BW_HALVING = 210000
    BLOCKS_PER_SEC = 1 / dt.timedelta(minutes=10).total_seconds()
    _TARGET_HASH = None

    def update_meta(self):
        url = 'https://blockchain.info/latestblock'
        block = requests.get(url).json()
        del block['txIndexes']

        block_url = f'https://blockchain.info/rawblock/{block["hash"]}'
        block = requests.get(block_url).json()
        del block['tx']

        # tgturl = 'https://learnmeabitcoin.com/technical/target'
        # tgthtml = BS(requests.get(tgturl).text, features="lxml")
        # tgtstr = tgthtml.find("code", {"class": "target"}).text
        # blkstr = tgthtml.find("small", {"class": "grey"}).text
        
        self._TARGET_HASH = decompress(hex(block['bits'])[2:])
        self._CURRENT_BLOCK = block['height']
        self.LAST_BLOCK_DETAILS = block

        self.CURRENT_PRICE_FOR_ALL_CURRENCIES = current_btc_price()
        self.last_update = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        assert self.DIFFICULTY_1_hash > self._TARGET_HASH

    @property
    def difficulty(self):
        if not hasattr(self, '_difficulty'):
            self._difficulty = self.difficulty_from_tgt()

        return self._difficulty

    def set_difficulty(self, d=None):
        if d is None:
            self._difficulty = self.difficulty_from_tgt()
        else:
            self._difficulty = d

    @property
    def TARGET_HASH(self):
        return self._TARGET_HASH

    @property
    def CURRENT_BLOCK(self):
        return self._CURRENT_BLOCK

    @property
    def CURRENT_PRICE(self):
        return self.CURRENT_PRICE_FOR_ALL_CURRENCIES.USD.loc['last']

    def historical_transaction_fees(self, *args, reload=True, **kwargs):
        if reload:
            try:
                url = blockchain_api('transaction-fees', *args, **kwargs)
                df = pd.read_csv(url, header=None, index_col=0, parse_dates=[0])
                df.index.name = 'Date'
                df.columns = ['Fees']
                self._historical_transaction_fees = df.Fees
            except Exception as e:
                if not hasattr(self, '_historical_transaction_fees'):
                    raise e

        return self._historical_transaction_fees

    def historical_difficulty(self, *args, reload=True, **kwargs):
        if reload:
            try:
                url = blockchain_api('difficulty', timespan='10years', *args, **kwargs)
                df = pd.read_csv(url, header=None, index_col=0, parse_dates=[0])
                df.index.name = 'Date'
                df.columns = ['Difficulty']
                self._historical_difficulty = df.Difficulty
            except Exception as e:
                if not hasattr(self, '_historical_difficulty'):
                    raise e
        
        return self._historical_difficulty

    def historical_price(self, *args, reload=True, **kwargs):
        if reload:
            try:
                df = pd.read_csv(blockchain_api('market-price', timespan='10years', *args, **kwargs), header=None, index_col=0, parse_dates=[0])
                df.index.name = 'Date'
                df.columns = ['Price']
                self._historical_price = df.Price
            except Exception as e:
                if not hasattr(self, '_historical_price'):
                    raise e

        return self._historical_price

    def historical_revenue(self, *args, reload=True, **kwargs):
        if reload:
            try:
                df = pd.read_csv(blockchain_api('miners-revenue', timespan='10years', *args, **kwargs), header=None, index_col=0, parse_dates=[0])
                df.index.name = 'Date'
                df.columns = ['Revenue']
                self._historical_revenue = df.Revenue
            except Exception as e:
                if not hasattr(self, '_historical_revenue'):
                    raise e
        return self._historical_revenue

    def historical_hash_rate(self, *args, reload=True, **kwargs):
        if reload:
            try:
                df = pd.read_csv(blockchain_api('hash-rate', timespan='10years', *args, **kwargs), header=None, index_col=0, parse_dates=[0])
                df.index.name = 'Date'
                df.columns = ['Hash Rate']
                self._historical_hash_rate = df['Hash Rate']
            except Exception as e:
                if not hasattr(self, '_historical_hash_rate'):
                    raise e            
        return self._historical_hash_rate

    def historical_values(self, *args, **kwargs):
        return pd.concat([
            self.historical_price(*args, **kwargs),
            self.historical_revenue(*args, **kwargs),
            self.historical_difficulty(*args, **kwargs),
            self.historical_hash_rate(*args, **kwargs),
            self.historical_transaction_fees(*args, **kwargs)
        ], axis=1)

    def check_meta(self):
        url = 'https://bitcoinblockhalf.com/'

        tables = pd.read_html(requests.get(url).text)

        df_btc = tables[0]
        df_btc = df_btc.set_index(0)
        df_btc.index = df_btc.index.str.replace(':', '')
        df_btc.index.name = 'index'
        df_btc.index = df_btc.index.str.lower().str.replace(' ', '_')
        df_btc = df_btc.squeeze()

        def fix_type(val):
            try:
                return float(val)
            except:
                if '%' in val:
                    return float(val.replace('%', '')) / 100
                elif '$' in val:
                    return float(val.replace('$', '').replace(',', ''))
                elif ' ' in val:
                    return float(val.split(' ')[0])
                else:
                    return val

        df_btc = df_btc.apply(fix_type)

        return df_btc

    @property
    def CURRENT_EPOCH(self):
        return (self.halving_blocks() < self.CURRENT_BLOCK).sum()

    @property
    def CURRENT_BLOCK_REWARD(self):
        return self.reward_increments()[self.CURRENT_EPOCH - 1]

    def retgt_blocks(self, epochs=1000):
        return np.arange(0, self.RETARGET_BLOCKS*epochs, self.RETARGET_BLOCKS)

    def block_periods_from_now(self, *args, **kwargs):
        return pd.period_range(start=dt.datetime.now(), periods=self.until_halving(*args, **kwargs)[-1], freq='10min')
    
    def period_of_last_block(self): 
        return self.block_periods_from_now()[-1]

    def halving_blocks(self, epoch=9, end_block=False):
        # Returns either the initial block of each epoch or the ending block
        if end_block:
            epoch += 1

        halvings = np.arange(0, self.BLOCKS_BW_HALVING*epoch, self.BLOCKS_BW_HALVING)

        if end_block:
            halvings = halvings[1:] - 1

        return halvings

    def until_halving(self, *args, **kwargs):
        return self.halving_blocks(*args, **kwargs) - self.CURRENT_BLOCK

    def until_next_halving(self, *args, **kwargs):
        halvings = self.until_halving(*args, **kwargs)
        return halvings[halvings > 0][0]

    def reward_increments(self, epoch=9):
        arr = np.ones(epoch)
        arr[1:] = np.repeat(.5, arr.size - 1).cumprod()
        return self.INIT_BLOCK_REWARD * arr

    def reward_schedule(self, epoch=100):
        sched = pd.Series(self.reward_increments(epoch=epoch), index=self.halving_blocks(epoch=epoch), name='reward')
        sched.index.name = 'block_id'
    
        return sched

    def time_to_retarget(self, blocks):
        return pd.Timedelta(blocks / self.BLOCKS_PER_SEC, units='s')

    def generate_block_schedule(self, start=None, end_of_epoch=None, epochs_ahead=None):
        if all((end_of_epoch is None, epochs_ahead is None)):
            raise ValueError('You must provide one of `end_of_epoch` or `epochs_ahead`')

        if epochs_ahead is not None:
            end_of_epoch = self.CURRENT_EPOCH + epochs_ahead

        if end_of_epoch < self.CURRENT_EPOCH:
            raise ValueError('`epoch` cannot be less than the current epoch {self.CURRENT_EPOCH}')

        return BlockSchedule(self, start, end_of_epoch)

    def summary(self):
        return pd.Series([
            self.last_update,
            self.CURRENT_PRICE,
            self.CURRENT_BLOCK,
            self.CURRENT_EPOCH,
            self.CURRENT_BLOCK_REWARD,
            self.TARGET_HASH,
            self.difficulty,
            self.expected_hash_rate().__repr__(),
            ],
            index=['Last Updated', 'Price', 'Block', 'Epoch', 'Reward', 'Target', 'Difficulty', 'Hash Rate'], name='Summary'
        )

class BlockSchedule:
    def __new__(cls, BTC, start=None, end_of_epoch=9):
        reward_schedule = BTC.reward_schedule(epoch=end_of_epoch)

        block_periods = BTC.block_periods_from_now(end_of_epoch, end_block=True)
        block_schedule = pd.Series(block_periods, name='period')
        block_schedule.index = block_schedule.index + BTC.CURRENT_BLOCK
        block_schedule.index.name = 'block_id'

        block_schedule = block_schedule.to_frame()
        block_schedule.loc[:, 'reward'] = np.nan
        
        icurr = reward_schedule.index.get_indexer([BTC.CURRENT_BLOCK], method='ffill')[0]
        block_schedule.loc[block_schedule.index[0], 'reward'] = reward_schedule.iloc[icurr]

        for block_id, reward in reward_schedule.iloc[icurr + 1:].iteritems():
            block_schedule.loc[block_id, 'reward'] = reward

        block_schedule.reward = block_schedule.reward.ffill()

        block_schedule = block_schedule.reset_index().set_index('period')

        if start is not None:
            block_schedule = block_schedule.loc[start:]

        # Add retarget block indicators
        retgts = np.intersect1d(block_schedule.block_id.values, BTC.retgt_blocks())
        block_schedule.loc[:, 'retarget'] = block_schedule.block_id.isin(retgts)

        return block_schedule
