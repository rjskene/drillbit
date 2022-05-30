from pathlib import Path
from anyio import wrap_file
import pandas as pd
import xlwings as xw
from xlwings.utils import rgb_to_int

import sys
root = Path('C:\\Users\\Ryan.Skene\\code\\finstat\\')
btc_path = root / '_private' / 'mara' / 'bitcoin'

sys.path.insert(0, root.as_posix())
sys.path.insert(0, btc_path.as_posix())

from examples.btc.objects import calc_network_hash_rate,  MiningProfile
from examples.btc.units import EnergyCost, Power

from bitcoin import price_lookup, GRCStyle

style = GRCStyle()

COMPSHEET = 'Snapshot'
wb_btc = xw.Book(btc_path / 'snapshot.xlsm')

miners = wb_btc.sheets['Miners'].range('MinerDB[[#All]]').options(pd.DataFrame, index=False).value
coolers = wb_btc.sheets['Cooling'].range('CoolingProfiles[[#All]]').options(pd.DataFrame, index=False).value

mining_series_ilocs = [0,1,2,-2,-1]
miner_series_ilocs = [0,5,6,8,9,10]
miner_row = 11
cooling_row = 18

def upsample(val, freq):
    if freq == 'Month':
        val *= 24 * 6 * 30
    if freq == 'Annual':
        val *= 24 * 6 * 365

    return val

class Profiles:
    miners = wb_btc.sheets['Miners'].range('MinerDB[[#All]]').options(pd.DataFrame, index=False).value
    coolers = wb_btc.sheets['Cooling'].range('CoolingProfiles[[#All]]').options(pd.DataFrame, index=False).value

    def init(self):
        self.profiles = {}
        self.miners = wb_btc.sheets['Miners'].range('MinerDB[[#All]]').options(pd.DataFrame, index=False).value
        self.coolers = wb_btc.sheets['Cooling'].range('CoolingProfiles[[#All]]').options(pd.DataFrame, index=False).value

    def assign_profile(self, **kwargs):
        kwargs['energy_cost'] = EnergyCost(kwargs['energy_cost'], 'MWh')
        kwargs['power'] = Power(kwargs['power'], 'MW')
        kwargs['miner'] = self.miners.miners.get_by_name(kwargs['miner'])
        kwargs['cooling'] = self.coolers.coolers.get_by_name(kwargs['cooling'])
        self.profiles[kwargs['name']] = MiningProfile(**kwargs)

PROFILES = Profiles()

@xw.sub
def create_profiles():
    wb_btc.sheets[COMPSHEET].range('G6:I6').clear_contents()
    inputs = wb_btc.sheets[COMPSHEET].range('A2:D20').options(pd.DataFrame, index=True, header=False).value

    cleaned = inputs.loc['Project':].iloc[1:4].copy()
    cleaned.loc['Miner'] = inputs.loc['Miner':].iloc[1]
    cleaned.loc['Cooling'] = inputs.loc['Cooling':].iloc[1]
    cleaned = cleaned.T[cleaned.notna().all(axis=0)].T
    cleaned.index = cleaned.index.str.split().str.join('_').str.lower()

    PROFILES.init()
    for idx, col in cleaned.iteritems():
        PROFILES.assign_profile(**col)

    for i, name in enumerate(list(PROFILES.profiles.keys())):
        wb_btc.sheets[COMPSHEET].range('G6:I6')[i].value = name

    wb_btc.sheets[COMPSHEET].api.OLEObjects('CommandButton1').Object.BackColor = rgb_to_int(style.vvlblue.hex_to_rgb())
    wb_btc.sheets[COMPSHEET].api.OLEObjects('CommandButton1').Object.Caption = 'Updated!'

@xw.func
def mining_series(name, index:int, with_name=1):
    index = bool(index)
    if name is not None and hasattr(PROFILES, 'profiles') and name in PROFILES.profiles:
        prof = PROFILES.profiles[name]
        ser = prof.as_series(as_repr=True).str.strip("'").iloc[mining_series_ilocs]
        if not with_name:
            ser = ser.iloc[3:]
        if index:
            return ser.values.reshape(-1,1)
        else:
            return ser
    else:
        return None

@xw.func
def miner_series(name, index:int, with_name=1, project_name=None):
    # project_name is purely retrigger equation on profile update
    with_name = bool(with_name)
    index = bool(index)
    if name is not None:
        prof = PROFILES.miners.miners.get_by_name(name)
        ser = prof.as_series(as_repr=True).str.strip("'").iloc[miner_series_ilocs]
        if not with_name:
            ser = ser.iloc[1:]
        if index:
            return ser.values.reshape(-1,1)
        else:
            return ser
        return None

@xw.func
def cooling_series(name, index:int, with_name=True, project_name=None):
    # project_name is purely retrigger equation on profile update
    index = bool(index)
    if name is not None:
        prof = PROFILES.coolers.coolers.get_by_name(name)
        ser = prof.as_series(as_repr=True).str.strip("'").iloc[:-1]
        if not with_name:
            ser = ser.iloc[1:]
        if index:
            return ser.values.reshape(-1,1)
        else:
            return ser
    else:
        return None

@xw.func
def expected_hash_rate(difficulty):
    return calc_network_hash_rate(difficulty).__repr__()

@xw.func
def block_win_per(name, difficulty):
    if name and hasattr(PROFILES, 'profiles') and name in PROFILES.profiles:
        prof = PROFILES.profiles[name]
        return prof.likelihood_per_block(difficulty=difficulty)

@xw.func
def block_revenue(name, freq, *args):
    if name and hasattr(PROFILES, 'profiles') and name in PROFILES.profiles:
        args = [float(arg) for arg in args]
        prof = PROFILES.profiles[name]
        return upsample(prof.revenue_per_block(*args), freq)

@xw.func
def block_cogs(name, freq):
    if name and hasattr(PROFILES, 'profiles') and name in PROFILES.profiles:
        prof = PROFILES.profiles[name]
        return upsample(prof.cogs_per_block(), freq)

@xw.func
def block_miner_cost(name, freq):
    if name and hasattr(PROFILES, 'profiles') and name in PROFILES.profiles:
        prof = PROFILES.profiles[name]
        return upsample(prof.cost_of_miners_per_block(), freq)

@xw.func
def total_miner_cost(name):
    if name and hasattr(PROFILES, 'profiles') and name in PROFILES.profiles:
        prof = PROFILES.profiles[name]
        return prof.cost_of_miners()

@xw.func
def block_cooling_cost(name, freq):
    if name and hasattr(PROFILES, 'profiles') and name in PROFILES.profiles:
        prof = PROFILES.profiles[name]
        return upsample(prof.cost_of_cooling_per_block(), freq)

@xw.func
def total_cooling_cost(name):
    if name and hasattr(PROFILES, 'profiles') and name in PROFILES.profiles:
        prof = PROFILES.profiles[name]
        return prof.cost_of_cooling()

@xw.func
def block_capital_cost(name, freq):
    if name and hasattr(PROFILES, 'profiles') and name in PROFILES.profiles:
        prof = PROFILES.profiles[name]
        return upsample(prof.capital_cost_per_block(), freq)

@xw.func
def total_capital_cost(name):
    if name and hasattr(PROFILES, 'profiles') and name in PROFILES.profiles:
        prof = PROFILES.profiles[name]
        return prof.capital_cost()

def find_hashes(prof, period):
    if period == 'Block' or None:
        hashes = prof.hash_rate.hashes_per_block()
    elif period == 'Month':
        hashes = prof.hash_rate.hashes_per_month()
    elif period == 'Annual':
        hashes = prof.hash_rate.hashes_per_year()

    return hashes

@xw.func
def hashes(name, period):
    if name and hasattr(PROFILES, 'profiles') and name in PROFILES.profiles:
        prof = PROFILES.profiles[name]
        hashes = find_hashes(prof, period).ZH
        return hashes.scaled(hashes.magnitude)

@xw.func
def price_per_hash(name, price, period):
    if name and hasattr(PROFILES, 'profiles') and name in PROFILES.profiles:
        price = float(price)
        prof = PROFILES.profiles[name]
        hashes = find_hashes(prof, period)
        return prof.price_per_hash(price, hashes).EH.__repr__().split(' ')[0]

@xw.func
def price_per_power(name, price):
    if name and hasattr(PROFILES, 'profiles') and name in PROFILES.profiles:
        price = float(price)
        prof = PROFILES.profiles[name]
        return prof.price_per_power(price).MW.__repr__().split(' ')[0]

