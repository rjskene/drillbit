"""
### TO DO ###

> end_of_epoch value is not being set correctly; it returns the beginning of NEXT
epoch, not of provided epoch. so if `current_epoch=4`, if `end_of_epoch=4` is provided,
it does not generate a schedule
"""

import requests
import datetime as dt
import numpy as np
import pandas as pd
import requests

from .mining import MiningMixin, decompress

# Utility functions for modelling price, fees, and other network metrics
def compound_growth(init, g, n):
    arr = np.ones(n)
    arr[1:] += g
    return init * arr.cumprod()

def geometric_brownian_motion(S0, n, T, mean=0.01, volatility=0.01):
    mu = mean
    sigma = volatility
    dt = T / n
    t = np.arange(n)
    W = np.random.standard_normal(size=n)
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    X = (mu - 0.5*sigma**2)*t + sigma*W
    return S0*np.exp(X)

# Blockchain.info API client
class BlockChainAPIClient:
    # $chartName?timespan=$timespan&rollingAverage=$rollingAverage&start=$start&format=$format&sampled=$sampled'
    root = 'https://api.blockchain.info/charts/{chartName}?timespan={timespan}&format={fmt}'

    def _get_then_set(self, chartName, timespan='10years', fmt='csv'):
        url = self.root.format(chartName=chartName, timespan=timespan, fmt=fmt)
        df = pd.read_csv(url, header=None, index_col=0, parse_dates=[0])
        df.index.name = 'Date'
        df.columns = [chartName]
        setattr(self, '_historical_' + chartName.replace('-', '_'), df[chartName])

        return getattr(self, '_historical_' + chartName.replace('-', '_'))

    def current_prices(self):
        return pd.read_json('https://blockchain.info/ticker')

    def latest_block(self, raw=True):
        url = 'https://blockchain.info/latestblock'
        block = requests.get(url).json()        
        del block['txIndexes']

        if raw:
            block_url = f'https://blockchain.info/rawblock/{block["hash"]}'
            block = requests.get(block_url).json()
            del block['tx']
        
        return block

    def historical_transaction_fees(self, *args, reload=True, **kwargs):
        if reload:
            self._get_then_set('transaction-fees', *args, **kwargs)

        return self._historical_transaction_fees

    def historical_difficulty(self, *args, reload=True, **kwargs):
        if reload:
            self._get_then_set('difficulty', *args, **kwargs)

        return self._historical_difficulty

    def historical_price(self, *args, reload=True, **kwargs):
        if reload:
            self._get_then_set('market-price', timespan='10years', *args, **kwargs)

        return self._historical_market_price

    def historical_revenue(self, *args, reload=True, **kwargs):
        if reload:
            self._get_then_set('miners-revenue', timespan='10years', *args, **kwargs)
        
        return self._historical_revenue

    def historical_hash_rate(self, *args, reload=True, **kwargs):
        if reload:
            self._get_then_set('hash-rate', *args, **kwargs)
        return self._historical_hash_rate

    def historical_values(self, *args, **kwargs):
        return pd.concat([
            self.historical_price(*args, **kwargs),
            self.historical_revenue(*args, **kwargs),
            self.historical_difficulty(*args, **kwargs),
            self.historical_hash_rate(*args, **kwargs),
            self.historical_transaction_fees(*args, **kwargs)
        ], axis=1)

class ProjectionUtilityMixin:
    def retgt_blocks(self, epochs=1000):
        return np.arange(0, self.RETARGET_BLOCKS*epochs, self.RETARGET_BLOCKS)

    def block_periods_from_now(self, *args, **kwargs):
        return pd.period_range(start=dt.datetime.now(), periods=self.until_halving(*args, **kwargs)[-1] + 1, freq='10min')
    
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
        return self.halving_blocks(*args, **kwargs) - self.current_block

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

    def generate_block_schedule(self, start=None, end_of_epoch=None, epochs_ahead=None, trim_last=True):
        if all((end_of_epoch is None, epochs_ahead is None)):
            raise ValueError('You must provide one of `end_of_epoch` or `epochs_ahead`')

        if epochs_ahead is not None:
            end_of_epoch = self.current_epoch + epochs_ahead

        if end_of_epoch < self.current_epoch:
            raise ValueError('`epoch` cannot be less than the current epoch {self.current_epoch}')

        if isinstance(start, dt.date):
            start = start.strftime('%Y-%m-%d')

        reward_schedule = self.reward_schedule(epoch=end_of_epoch)
                
        block_periods = self.block_periods_from_now(end_of_epoch, end_block=False)
        
        block_schedule = pd.Series(block_periods, name='period')
        block_schedule.index = block_schedule.index + self.current_block
        block_schedule.index.name = 'block_id'

        block_schedule = block_schedule.to_frame()
        block_schedule.loc[:, 'reward'] = np.nan
        
        icurr = reward_schedule.index.get_indexer([self.current_block], method='ffill')[0]
        block_schedule.loc[block_schedule.index[0], 'reward'] = reward_schedule.iloc[icurr]

        for block_id, reward in reward_schedule.iloc[icurr + 1:].iteritems():
            block_schedule.loc[block_id, 'reward'] = reward

        block_schedule.reward = block_schedule.reward.ffill()

        block_schedule = block_schedule.reset_index().set_index('period')

        if start is not None:
            block_schedule = block_schedule.loc[start:]

        # Find halvings
        block_schedule.loc[:, 'halvings'] = block_schedule.reward.diff().shift(-1).fillna(0) != 0

        # Find retargets
        retgts = np.intersect1d(block_schedule.block_id.values, self.retgt_blocks())
        block_schedule.loc[:, 'retarget'] = block_schedule.block_id.isin(retgts)

        if trim_last:
            block_schedule = block_schedule.iloc[:-1]

        return block_schedule

    def constant(self, initial, n):
        return np.ones(n) * initial

    def cgr(self, initial, n, g):
        return compound_growth(initial, g, n)

    def gbm(self, initial, periods, *args, **kwargs):
        n = periods.size
        if periods.dtype == 'object':
            start = pd.Period(periods.iloc[0], freq='10min')
            end = pd.Period(periods.iloc[-1], freq='10min')
        else:
            start = periods.iloc[0]
            end = periods.iloc[-1]

        T = (end - start).delta.days
        return geometric_brownian_motion(initial, n, T, *args, **kwargs)

    def forecast(self, model, periods, initial, mean=None, volatility=None):
        ALLOWED_MODELS = ['Constant', 'CGR', 'GBM']

        if model not in ALLOWED_MODELS:
            raise ValueError(f'{model} is not acceptable model type')

        if model == 'CGR' and mean is None:
            raise ValueError('`mean` must be provided for CGR model')
        
        if model == 'GBM' and any((mean is None, volatility is None)):
            raise ValueError('`mean` and `volatility` must be provided for GBM model')

        n = periods.size
        if model == 'Constant':
            forecast = self.constant(initial, n)
        elif model == 'CGR':
            forecast = self.cgr(initial, n, mean)
        elif model == 'GBM':
            forecast = self.gbm(initial, periods, mean, volatility)
        
        return pd.DataFrame({'period': periods, 'forecast': forecast})

class BitcoinEnvironmentUtility(ProjectionUtilityMixin, MiningMixin):
    INIT_BLOCK_REWARD = 50
    RETARGET_BLOCKS = 2016
    BLOCKS_BW_HALVING = 210000
    BLOCKS_PER_SEC = 1 / dt.timedelta(minutes=10).total_seconds()
    _TARGET_HASH = None

    def __init__(self):
        self.client = BlockChainAPIClient()
        self.update_meta()
        # self.mock_meta()

    def mock_meta(self):
        self._CURRENT_BLOCK = 712000        

    def update_meta(self):
        print ('updating meta', end='...')
        self._CURRENT_BLOCK_DETAILS = self.client.latest_block()
    
        self._TARGET_HASH = decompress(hex(self._CURRENT_BLOCK_DETAILS['bits'])[2:])
        self._CURRENT_BLOCK = self._CURRENT_BLOCK_DETAILS['height']

        self.current_prices = self.client.current_prices()
        self.last_update = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        assert self.DIFFICULTY_1_hash > self._TARGET_HASH
        print ('done')

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
    def target_hash(self):
        return self._TARGET_HASH

    @property
    def current_block(self):
        return self._CURRENT_BLOCK

    @property
    def current_price(self):
        return self.current_prices.USD.loc['last']

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
    def current_epoch(self):
        return (self.halving_blocks() < self.current_block).sum()

    @property
    def current_reward(self):
        return self.reward_increments()[self.current_epoch - 1]

    def summary(self):
        return pd.Series([
            self.last_update,
            self.current_price,
            self.current_block,
            self.current_epoch,
            self.current_reward,
            self.target_hash,
            self.difficulty,
            self.expected_hash_rate().__repr__(),
            ],
            index=['Last Updated', 'Price', 'Block', 'Epoch', 'Reward', 'Target', 'Difficulty', 'Network Hash Rate'], name='Summary'
        )