from pathlib import Path
import datetime as dt
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
import dill as pickle
import yaml
import xlwings as xw

from btc.objects import CoolingProduct, CoolingProducts, CoolingProfiles, Miners, MiningProfiles
from btc.meta import init_meta

from .helpers import fs_read_pickle, fs_write_pickle
from .excel import DataFrameDropna

def create_product(xl):
    costs = xl.parse('Cost Schedule').dropna()
    details = xl.parse('Details').T.reset_index().T.set_index(0).iloc[:, 0].fillna('')
    details.index = details.index.str.lower().str.replace(' ', '_')
    details.index = details.rename(index={'number_of_miners': 'n_miners'}).index

    if details.dimensions:
        details.dimensions = tuple([float(d) for d in details.dimensions.split(', ')])

    if details.options:
        details.options = details.options.split(', ')

    details.loc['cost_schedule'] = costs

    return CoolingProduct(**details.to_dict())

class SheetState:
    LOAD_ATTRS  = dict(
        fstats = ['env'],
        objs = [
            'cooling_products', 
            'block_sched', 
            'periods',
            'btc_price', 
            'traxn_fees',
        ],
        dfs = ['miners', 'coolers', 'pools'],
    )

    def __init__(self, exec_file, BTC=None, config=None):
        wb_name = exec_file.split('\\')[-1].split('.')[0] # get file name; must correspond to spreadsheet

        self.wb = xw.books[f'{wb_name}.xlsm']
        self.parent_path = Path(exec_file).parent.resolve()
        self.BTC = BTC

        self.save_config(config)

    def save_config(self, config=None):
        if config is None:
            config = self.parent_path / 'config.yml'
            with config.open() as c:
                config = yaml.safe_load(c)

            if 'config' in [n.name for n in self.wb.sheets]:
                config_xl = self.get_config()
                config = config | config_xl

        for k, v in config.items():
            setattr(self, k, v)

    def get_config(self):
        config_xl = self.wb.sheets['config'].range('A1').expand().options(pd.DataFrame, header=False).value.loc[:, 0]
        config_xl.index = config_xl.index.str.replace(' ', '_')
        return config_xl.to_dict()

    def update_config(self):
        for k, v in self.get_config().items():
            setattr(self, k, v)

    @property
    def WB_NAME(self):
        return self.wb.name.split('.')[0]

    @property
    def WS(self):
        return self.WORKSHEETS

    @property
    def datadir(self):
        if hasattr(self, 'data_path'):
            return self.parent_path.parent / self.data_path
        else:
            return self.parent_path.parent / 'data'

    @property
    def mines(self):
        if hasattr(self, 'projects'):
            return self.pools + self.projects
        else:
            return self.pools

    def get_ws(self, ws):
        return self.wb.sheets[self.WS[ws]]

    def get_table(self, ws, tbl):
        return self.get_ws(ws).range(self.TABLES[tbl])

    def is_implemented(self):
        return self.implemented

    def has_schedule(self):
        return hasattr(self, 'block_sched')

    def has_btc_forecast(self):
        return hasattr(self, 'btc_price')

    def has_traxn_fee_forecast(self):
        return hasattr(self, 'traxn_fees')

    def has_cooling(self):
        return hasattr(self, 'coolers')

    def has_miners(self):
        return hasattr(self, 'miners')

    def has_mines(self):
        return hasattr(self, 'mines')

    def set_implemented(self, value):
        self.implemented = value

    def set_global_environment(self, stat):
        self.env = stat

    def set_mine_statements(self, minestats):
        self.minestats = minestats

    def set_project_statements(self, projstats):
        self.projstats = projstats

    def update_btc(self):
        try:
            self.BTC = init_meta()
            date = dt.datetime.now().strftime('%Y-%m-%d')
            with open(self.datadir / 'meta' / f'meta-{date}.pkl', 'wb') as file_:
                pickle.dump(self.BTC, file_)
        except Exception as e:
            print ('There was an error updating the bitcoin meta data. Loading from prior version instead.', end='\n')
            print (repr(e))
            with open(self.datadir / 'meta' / 'meta-16-6-22.pkl', 'rb') as file_:
                self.BTC = pickle.load(file_)

        self.wb.sheets[self.WS['meta']].range(self.TABLES['meta']).options(index=False, header=False).value = self.BTC.summary().to_frame().reset_index()

    def update_miners(self):
        obj = self.wb.sheets[self.WS['miners']].range(f'{self.TABLES["miners"]}[[#All]]').options(DataFrameDropna, index=False).value
        self.miners = Miners(obj)

    def update_cooling(self):
        datadir = self.datadir / 'cooling'
        self.cooling_products = CoolingProducts([create_product(pd.ExcelFile(xlpath)) for xlpath in datadir.iterdir()])
        
        ws = self.get_ws('cooling')
        prod_col = ws.range(self.TABLES['cooling'] + '[Product]')
        prod_col.api.Validation.Delete()
        prod_col.api.Validation.Add(Type=3,Formula1=','.join(self.cooling_products.names))

        obj = ws.range(self.TABLES['cooling'] + '[[#All]]').options(DataFrameDropna, index=False).value
        self.coolers = CoolingProfiles(obj, self.cooling_products)

    def clean_mine_profiles(self, obj):
        obj.Overclock /= 100
        impl_keys = ['Direction', 'Start', 'Completion', 'Amount']
        impl_obj = obj.loc[:, impl_keys]
        impl_obj.columns = impl_obj.columns.str.lower()
        obj.loc[:, 'impl_kws'] = impl_obj.to_dict('records')
        obj = obj.loc[:, ~obj.columns.isin(impl_keys)]
        
        return obj

    def update_pools(self):
        obj = self.get_ws('pools').range(f'{self.TABLES["pools"]}[[#All]]').options(DataFrameDropna, index=False).value
        self.pools = MiningProfiles(self.clean_mine_profiles(obj), miners=self.miners, coolers=self.coolers, power='GW', opex_cost='GW', density='kW')

    def update_projects(self):
        obj = self.get_ws('projects').range(f'{self.TABLES["projects"]}[[#All]]').options(DataFrameDropna, index=False).value
        self.projects = MiningProfiles(self.clean_mine_profiles(obj), miners=self.miners, coolers=self.coolers, power='GW', opex_cost='GW', density='kW')

    def set_block_sched(self, block_sched):
        self.block_sched = block_sched
        self.periods = block_sched.index

    def set_btc_price(self, price):
        btc_price = pd.Series(price, index=self.periods)
        btc_price.index.name = 'Period'
        self.btc_price = btc_price

    def price_cagrs(self):
        btc = self.btc_price.resample('M').last().values
        ns = [36, 60, btc.size]
        vals = [(btc[:n][-1] / btc[0])**(1/(n/12)) - 1 for n in ns]
        index = ['3-YR', '5-YR', f'{btc.size/12:.1f}-YR']
        return pd.Series(vals, index=index)

    def set_traxn_fees(self, fees):
        fees = pd.Series(fees, index=self.periods)
        fees.index.name = 'Period'
        self.traxn_fees = fees
    
    def generate_block_schedule(self):
        start_date, epoch = self.wb.sheets[self.WS['block_sched']].range(self.TABLES['block_sched_inputs']).value

        sched = self.BTC.generate_block_schedule(start_date.strftime('%Y-%m-%d'), int(epoch))
        self.set_block_sched(sched)

        sched = sched.copy()
        sched.columns = sched.columns.str.split('_').str.join(' ').str.title()
        sched.index.name = 'Period'
        return sched

    def _forecast_by_model(self, model, init, mu, sigma):
        ALLOWED_MODELS = ['Constant', 'CGR', 'GBM']

        if model not in ALLOWED_MODELS:
            raise ValueError(f'{model} is not acceptable model type')

        if model == 'Constant':
            forecast = init * np.ones(self.periods.size)
        elif model == 'CGR':
            forecast = self.BTC.cgr(init, self.periods.size, mu)
        elif model == 'GBM':
            forecast = self.BTC.gbm(init, self.periods, mu, sigma)

        return forecast

    def generate_btc_forecast(self):
        init_price, mu, sigma, model = self.get_ws('btc_price').range(self.TABLES['btc_price_inputs']).value
        price_forecast = self._forecast_by_model(model, float(init_price), mu, sigma)        
        self.set_btc_price(price_forecast)

        self.get_ws('btc_price').range(self.TABLES['btc_cagrs']).options(pd.Series).value = self.price_cagrs()

        return self.btc_price

    def generate_fee_forecast(self):
        init_price, mu, sigma, model = self.wb.sheets[self.WS['fees']].range(self.TABLES['fees_inputs']).value        
        fee_forecast = self._forecast_by_model(model, float(init_price), mu, sigma)
        self.set_traxn_fees(fee_forecast)

        return self.traxn_fees

    def implement_pools(self):
        self.pools.implement(self.periods.size)
        self.set_implemented(True)

    def implement_projects(self):
        self.projects.implement(self.periods.size)

    def wipe(self):
        if hasattr(self, 'projstats'):
            del self.projstats
        
        if hasattr(self, 'minestats'):
            del self.minestats

        import gc
        gc.collect()

    def load_environment(self, dirname):
        dirpath = self.datadir / 'envs' / dirname
        metapath = dirpath / 'meta'

        for k, attrs_to_load in self.LOAD_ATTRS.items():
            for attr in attrs_to_load:
                if k == 'fstats':
                    obj = fs_read_pickle(metapath / f'{attr}.pkl')
                else:
                    with open(metapath / f'{attr}.pkl', 'rb') as file1:
                        obj = pickle.load(file1)
                        if k == 'dfs':
                            with open(metapath / f'{attr}_units.pkl', 'rb') as file2:
                                units = pickle.load(file2) # each Profiles object must get units keywords to remain consistent
                            if attr == 'miners':  
                                obj = Miners(obj.drop(columns='index'), **units)
                            elif attr == 'coolers':
                                obj = CoolingProfiles(obj.drop(columns='index'), self.cooling_products, **units) # cooling_product should have already been loaded!
                            elif attr == 'pools':  
                                obj = MiningProfiles(obj.drop(columns='index'), miners=self.miners, coolers=self.coolers, **units)

                setattr(self, attr, obj)

    def save_environment(self, dirname):
        dirname += ' ' + dt.datetime.now().strftime('%y-%m-%d-%H-%M')
        dirpath = self.datadir / 'envs' / dirname
        if not dirpath.exists():
            dirpath.mkdir()

        metapath = dirpath / 'meta'
        if not metapath.exists():
            metapath.mkdir()

        assert not len(self.minestats[0].G.filter_graph_by_attribute('hidden', True).nodes)

        with tqdm(total=len(self.LOAD_ATTRS['dfs']) + len(self.LOAD_ATTRS['objs']) + len(self.LOAD_ATTRS['fstats']) + self.minestats.size + 1) as pbar:
            for fsstr in self.LOAD_ATTRS['fstats']:
                fs_write_pickle(getattr(self, fsstr), metapath / 'env.pkl')
                
            pbar.update(1)
                
            for dfstr in self.LOAD_ATTRS['dfs']:
                with open(metapath / f'{dfstr}.pkl', 'wb') as file_:
                    pickle.dump(getattr(self, dfstr).df, file_)
                
                with open(metapath / f'{dfstr}_units.pkl', 'wb') as file_:
                    pickle.dump(getattr(self, dfstr)._provided_units, file_) # All sublcasses of Profiles will have a `_provided_units` attr
                
                pbar.update(1)

            for objstr in self.LOAD_ATTRS['objs']:
                with open(metapath / f'{objstr}.pkl', 'wb') as file_:
                    pickle.dump(getattr(self, objstr), file_)

                pbar.update(1)

            envs = np.zeros((self.minestats.size, *self.minestats[0].env.shape))
            stats = np.zeros((self.minestats.size, *self.minestats[0].istat.shape))
            for i, mstat in enumerate(self.minestats):
                envs[i] = mstat.env.to_frame().values
                stats[i] = mstat.istat.to_frame().values

                pbar.update(1)

            with open(dirpath / 'stats.pkl', 'wb') as file_:
                pickle.dump(stats, file_)

            with open(dirpath / 'envs.pkl', 'wb') as file_:
                pickle.dump(envs, file_)

            with open(dirpath / 'sample_stats_frame.pkl', 'wb') as file_:
                pickle.dump(self.minestats[0].istat.to_frame(), file_)

            with open(dirpath / 'sample_envs_frame.pkl', 'wb') as file_:
                pickle.dump(self.minestats[0].env.to_frame(), file_)
                    
            pbar.update(1)
