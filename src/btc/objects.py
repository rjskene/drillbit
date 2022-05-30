from dataclasses import dataclass
import numpy as np
import pandas as pd

import finstat as fs

from .meta import MiningMixin
from .funcs import consumption_in_Wh
from .units import Power, Energy, HashRate, HashPrice, PowerPrice, \
    EnergyPrice, Efficiency, Density, Area, PropertyPrice

def _shorten(name, short_name=None) -> str:
    """"
    Logic for creating shorthand name for an LineItem object
    
    Examples:
        'Revenue' -> 'revenue'
        'Cost of Goods Sold' -> 'cogs'
        'Amortization' -> 'amort'

    Parameters:
        name: str
    """ 
    if short_name is None:
        if isinstance(name, (int, float)):
            return str(name)
        if name in ['', None]:
            return name
        elif name == 'Income Statement':
            return 'istat'
        elif len(name.split(' ')) > 1:
            return ''.join(s[0].lower() for s in name.split(' '))
        elif len(name) > 10:
            return name[:5].lower()
        else:       
            return name.lower()
    else:
        return short_name

class MinerPropertiesMixin:
    @property
    def pow(self):
        return self.power

    @property
    def eff(self):
        return self.efficiency

    @property
    def hr(self):
        return self.hash_rate

class Miner(MinerPropertiesMixin):
    """Class for keeping track of an item in inventory."""
    def __init__(self, 
        name: str, make: str, model: str, generation: str, manufacturer: str,
        price: float, 
        amortization: int=60, 
        short_name: str = None,
        power: Power = None, 
        hash_rate: HashRate = None, 
        efficiency: Efficiency = None,
        variance: float = 0, 
        overclock: float = 0,
        operate: bool = True,
        ):
        if pd.Series([power, hash_rate, efficiency]).isna().sum() > 1:
            raise ValueError('You must provide two of three of `power`, `hash_rate`, and `efficency`')

        self.name = name
        self.make = make
        self.model = model
        self.generation = generation
        self.manufacturer = manufacturer
        self.price = price
        self.amortization = amortization

        if short_name is None:
            short_name = self.name.replace(' ', '').lower()
        self.short_name = short_name

        self._specifications = Spec(power, hash_rate, efficiency, variance)
        self._operate = operate

        self.set_overclock(overclock)

    def __repr__(self):
        return f'Miner(name={self.name}, price={self.price}, ' \
        f'power={self.specs.power}, hash_rate={self.specs.hr}, efficiency={self.specs.eff}, ' \
        f'overclock={self.overclock}, operate={self._operate})'

    @property
    def amort(self):
        return self.amortization

    @property
    def specs(self):
        return self._specifications

    @property
    def OC(self):
        return self._overclock_mngr

    @property
    def power(self):
        if self.operating:
            if self.overclock:
                return self.OC.power_by_factor()
            else:
                return self.specs.power
        else:
            return Power(0)

    @property
    def hash_rate(self):
        if self.operating:
            if self.overclock:
                return self.OC.hash_rate_by_factor()
            else:
                return self.specs.hr
        else:
            return HashRate(0)

    @property
    def efficiency(self):
        if self.operating:
            if self.overclock:
                return self.OC.eff_by_factor()
            else:
                return self.specs.eff
        else:
            return Efficiency(0)

    def set_overclock(self, factor):
        self.overclock = bool(factor)
        if self.overclock:
            self._overclock_mngr = OverClock(factor=factor, miner=self)

    @property
    def operating(self):
        return self._operate

    def turn_off(self):
        self._operate = False

    def turn_on(self):
        self._operate = True

    def clone(self):
        kws = {k:v for k, v in self.__dict__.items() if '_' not in k}
        kws = kws | self.specs.__dict__
        return Miner(**kws)

    def consumption(self, **duration):
        return consumption_in_Wh(self.power, **duration)

    def consumption_variance(self, n, std=None):
        std = self.specs.variance / 2 if std is None else std
        return 1 + np.random.normal(0, std, n)

    def as_series(self, as_repr=False):
        ser = pd.Series(self.__dict__)

        ser.index = ser.index.str.split('_').str.join(' ').str.capitalize()

        if as_repr:
            ser.iloc[:] = [ser.__repr__() for ser in ser]
            
        return ser

class Spec(MinerPropertiesMixin):
    def __init__(self, power=None, hash_rate=None, efficiency=None, variance=0):
        self.power, self.hash_rate, self.efficiency = power, hash_rate, efficiency
        self.variance = variance

        if self.power is None:
            self.power = self.power_calc
        elif self.hash_rate is None:
            self.hash_rate = self.hr_calc
        elif self.efficiency is None:
            self.efficiency = self.eff_calc

    @property
    def power_calc(self):
        return self.efficiency * self.hr

    @property
    def hr_calc(self):
        return self.power / self.efficiency

    @property
    def eff_calc(self):
        return self.power / self.hr

class OverClock:
    """
    Governor for overclocking a miner    
    """
    a = 2.6696 # units of 1 / TH/s
    b = 23.33  # units of W / TH/s
    BASEMINER = Miner('Antminer S19', 'Antminer', 'S19', 'Base', 'Bitmain', 7000, 60, power=Power(3250), hash_rate=HashRate(96, 'TH'))
    
    def __init__(self, factor=0, miner=None, max_power=Power(6250), func=None):
        self._power_factor = factor
        self._miner = miner
        self.max_power = max_power
        self._func = func
        
    def base_func(self, x):
        # x = power in W
        # converts [W / TH/s] to [W / H/s] by taking in W
        # [1 / TH/s]*[TH/s][1e12H/s] * x W [1kW / 1000W] + [W / Th/s][Th/s / 1e12H/s]
        return ((self.a / 1e12) * x / 1000 + (self.b / 1e12)) / self.multiplier
    
    @property
    def func(self):
        return self.base_func if self._func is None else self._func
    
    @property
    def miner(self):
        return self.BASEMINER if self._miner is None else self._miner
    
    @property
    def multiplier(self):
        BASEMINER = object.__getattribute__(self, 'BASEMINER')
        miner = object.__getattribute__(self, 'miner')
        return BASEMINER.specs.efficiency / miner.specs.efficiency
    
    @property
    def power_factor(self):
        return self._power_factor
    
    def set_power_factor(self, factor):
        self._power_factor = factor

    @property
    def interval(self):
        return self.max_power - self.miner.specs.power
        
    def power_by_factor(self, n=None):
        if n is None:
            n = self._power_factor
        return Power((n * self.interval) + self.miner.specs.power)
        
    def curve(self, x=None):
        if x is None:
            x = np.linspace(self.miner.specs.power, self.max_power, 100)
        return x, self.func(x)
    
    def eff_by_power(self, x=None):
        if x is None:
            x = self.miner.specs.power
        return Efficiency(self.func(x))
    
    def eff_by_factor(self, n=None):
        return Efficiency(self.func(self.power_by_factor(n)))
    
    def hash_rate_by_power(self, x=None):
        if x is None:
            x = self.miner.specs.power
        return self.eff_by_factor(x).hash_rate(x)
    
    def hash_rate_by_factor(self, n=None):
        power = self.power_by_factor(n)
        return self.eff_by_factor(n).hash_rate(power)

@dataclass
class Cooling:
    """Class for keeping track of an item in inventory."""
    name: str
    capex: float
    pue: float
    style: str
    amortization: float = 60
    short_name: str = None

    def __post_init__(self):
        if self.short_name is None:
            self.short_name = self._shorten(self.name)

    def _shorten(self, *args, **kwargs):
        return _shorten(*args, **kwargs)

    @property
    def amort(self):
        return self.amortization

    def ancillary_consumption(self, energy):
        try:
            iteration = iter(energy)
        except TypeError:
            pass
        else:
            energy = np.array(energy) if not isinstance(energy, np.ndarray) else energy

        return Energy(energy * (self.pue - 1), abbr='Wh')

    def as_series(self, as_repr=False):
        ser = pd.Series(self.__dict__)
        ser.index = ser.index.str.split('_').str.join(' ').str.capitalize()

        if as_repr:
            ser.iloc[:] = [ser.__repr__() for ser in ser]
            
        return ser

@dataclass
class Mining(MiningMixin):
    name: str
    energy_cost: EnergyPrice
    power: Power
    pool_fee: float = 0
    short_name: str = None
    category: str = None
    miner: Miner = None
    cooling: Cooling = None
    opex_cost: PowerPrice = 0
    density: Density = 0
    property_cost: PropertyPrice = 1
    property_amort: int = 300
    property_value: PropertyPrice = 1
    property_tax_rate: float = 0
    overclock: float = 0
    impl_kws: dict = None
    
    def __post_init__(self):
        ALLOWED_CATEGORIES = ['Pool', 'Project', None]

        assert self.category in ALLOWED_CATEGORIES, self.category

        if self.short_name is None:
            self.short_name = self._shorten(self.name)
    
        if self.impl_kws:
            self.implement = Implementation(self, **self.impl_kws)

    def _shorten(self, *args, **kwargs):
        return _shorten(*args, **kwargs)
    
    def as_series(self, as_repr=False):
        dct = self.__dict__.copy()
        dct['Power - Miners'] = self.power_for_miners
        ser = pd.Series(dct)
        ser.miner = None if ser.miner is None else ser.miner.name
        ser.cooling = None if ser.cooling is None else ser.cooling.name
        ser.index = ser.index.str.split('_').str.join(' ').str.capitalize()
        ser.loc['# of Miners'] = self.n_miners

        if as_repr:
            ser.iloc[:] = [ser.__repr__() for ser in ser]

        return ser

    def assign_miner(self, miner, overclock=0):
        self.miner = miner
        if overclock:
            self.miner.set_overclock(overclock)

    def assign_cooling(self, cooling):
        self.cooling = cooling

    def halt(self):
        self.miner.turn_off()

    @property
    def power_for_miners(self):
        if self.cooling is None:
            return None
        else:
            return Power(self.power / self.cooling.pue)

    @property
    def power_for_cooling(self):
        if self.cooling is None:
            return None
        else:
            return Power(self.power - self.power_for_miners)

    @property
    def n_miners(self):
        if self.miner is None or self.cooling is None:
            return None
        elif not self.miner.operating:
            return 0
        else:
            return int(self.power_for_miners // self.miner.power)

    @property
    def hash_rate(self):
        return HashRate(self.n_miners * self.miner.hr)

    @property
    def is_project(self):
        return self.category == 'Project'

    @property
    def is_pool(self):
        return self.category == 'Pool'

    @property
    def cost_of_cooling(self):
        return self.power * self.cooling.capex

    @property
    def cost_of_miners(self):
        return self.n_miners * self.miner.price

    @property
    def footprint(self):
        return Area(self.power / self.density)

    @property
    def build_cost(self):
        return self.property_cost * self.footprint

    @property
    def property_valuation(self):
        return self.property_value * self.footprint

    @property
    def annual_property_taxes(self):
        return self.property_valuation * self.property_tax_rate

    @property
    def property_taxes_per_block(self):
        return self.annual_property_taxes / (365 * 24 * 6)

    @property
    def capital_cost(self):
        return self.cost_of_cooling + self.cost_of_miners + self.build_cost

    def cost_of_cooling_per_block(self, days_in_month=30):
        return self.cost_of_cooling / self.cooling.amort / (days_in_month * 24 * 6)

    def cost_of_miners_per_block(self, days_in_month=30):
        return self.cost_of_miners / self.miner.amort / (days_in_month * 24 * 6)

    def capital_cost_per_block(self):
        return self.cost_of_cooling_per_block() + self.cost_of_miners_per_block()

    def capital_cost_per_block(self, days_in_month):
        return self.cost_of_cooling_per_block(days_in_month) + self.cost_of_miners_per_block(days_in_month)

    def likelihood_per_block(self, *args, **kwargs):
        return super().likelihood_per_block(hash_rate=self.hash_rate, *args,  **kwargs)

    def cogs_per_block(self):
        return super().cogs_per_block(self.power, self.energy_cost)

    def price_per_hash(self, price, hashes):
        return HashPrice(price / hashes)

    def price_per_power(self, price):
        return PowerPrice(price / self.power)

class Implementation:
    ALLOWED_DIRECTIONS = ['Increasing', 'Declining', 'Stable']
    ALLOWED_COMPLETIONS = ['Last']
    
    def __init__(self, project, direction, completion, amount, start:int=0):
        assert direction in self.ALLOWED_DIRECTIONS, direction
        assert isinstance(completion, (float, int)) or completion in self.ALLOWED_COMPLETIONS, completion
        assert amount == 0 if direction == 'Increasing' else True, amount
        
        self.project = project
        self.direction = direction
        self._completion = completion if isinstance(completion, str) else int(completion)
        self.start = int(start)
        self.amount = amount
    
    def n_periods(self, n):
        if self._completion == 'Last':
            return n
        else:
            return self.months_to_blocks(self._completion)

    def months_to_blocks(self, n):
        return int(n * 30 * 24 * 6)

    def start_in_blocks(self):
        return self.months_to_blocks(self.start)

    def declining(self, n):
        end = self.project.n_miners * self.amount
        delta = np.linspace(self.project.n_miners, end, self.n_periods(n) - self.start_in_blocks()).astype('int')
        arr = np.ones(n)*end
        arr[self.start:delta.size] = delta
        return arr

    def increasing(self, n):
        delta = np.linspace(1, self.project.n_miners, self.n_periods(n) - self.start_in_blocks()).astype('int')
        arr = np.zeros(n)
        arr[self.months_to_blocks(self.start):delta.size + self.months_to_blocks(self.start)] = delta
        arr[delta.size + self.months_to_blocks(self.start):] = delta[-1]
        return arr

    def stable(self, n):
        return np.ones(n)*self.project.n_miners
    
    def implement(self, n):
        direction = self.direction.lower()
        return getattr(self, direction)(n)
    
    def __call__(self, n):
        self.project.miner_schedule = self.implement(n)
        return self.project

class Profiles(np.ndarray):
    _constructor = None

    class converters:
        @classmethod
        @property
        def attrs(cls):
            return {k: v for k, v in cls.__dict__.items() if '__' not in k and k != 'attrs'}

    @staticmethod
    def _construct_objs(cls, df, **units):
        objs = []
        for kws in list(df.T.to_dict().values()):
            for k, v in cls.converters.attrs.items():
                if k in kws:
                    klass, unit = v
                    if k in units:
                        unit = units[k]
                    kws[k] = klass(kws[k], unit)
            obj = cls._constructor(**kws)
            objs.append(obj)

        return objs

    def __new__(cls, df:pd.DataFrame, **units):
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        objs = cls._construct_objs(cls, df, **units)

        obj = np.asarray([o for o in objs], dtype='object').view(cls)
        
        if 'short_name' not in df.columns:
            df['short_name'] = [o.short_name for o in objs]
        df = df.reset_index().set_index('short_name')
        
        obj.df = df
        obj._provided_units = units

        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.df = getattr(obj, 'df', None)
        self._provided_units = getattr(obj, '_provided_units', None)

    def __getattribute__(self, name):
        if name == '__array_finalize__':
            return super().__getattribute__(name)
        else:
            df = object.__getattribute__(self, 'df')
            if name in df.index:
                iloc = df.loc[name, 'index']
                return self[iloc]
            elif hasattr(Mining, name):
                try:
                    return super().__getattribute__(name)
                except AttributeError as e:
                    try:
                        names = object.__getattribute__(self, 'names')
                        return pd.Series([getattr(p, name) for p in self], index=names, name=name)
                    except AttributeError as e:
                        raise e
                    except Exception as e:
                        raise e
            else:
                return super().__getattribute__(name)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.df.name.values:
                iloc = self.df[self.df.name == item]['index'].iloc[0]
                return self[iloc]
        return super().__getitem__(item)

    @property
    def names(self):
        return np.array([e.name for e in self])

    @property
    def short_names(self):
        return np.array([e.short_name for e in self])
    
class Miners(Profiles):
    _constructor = Miner

    class converters(Profiles.converters):
        hash_rate = (HashRate, 'TH')
        power = (Power, 'kW')

    def __repr__(self):
        joined = ', '.join([s.name for s in self])
        return f'Miners([{joined}])'

class CoolingProfiles(Profiles):
    _constructor = Cooling

    def __repr__(self):
        joined = ', '.join([s.name for s in self])
        return f'CoolingProfiles([{joined}])'

class MiningProfiles(Profiles):
    _constructor = Mining

    class converters(Profiles.converters):
        energy_cost = (EnergyPrice, 'MWh')
        power = (Power, 'MW')
        opex_cost = (PowerPrice, 'MW')
        density = (Density, 'W')
        property_cost = (PropertyPrice, 'sf')
        property_value = (PropertyPrice, 'sf')
    
    def __new__(cls, df:pd.DataFrame, miners:Miners=None, coolers:CoolingProfiles=None, **units):
        obj = super().__new__(cls, df, **units)
        
        if miners is not None:
            for profile in obj:
                profile.assign_miner(miners[profile.miner].clone())
                if profile.overclock:
                    profile.miner.set_overclock(profile.overclock)
        
        obj.miners = miners

        if coolers is not None:
            for profile in obj:
                profile.assign_cooling(coolers[profile.cooling])
        
        obj.coolers = coolers

        return obj
    
    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self.miners = getattr(self, 'miners', None)
        self.coolers = getattr(self, 'coolers', None)

    def __repr__(self):
        joined = ', '.join([s.name for s in self])
        return f'MiningProfiles([{joined}])'

    def __add__(self, item):
        df = pd.concat((self.df, item.df))
        df = df.reset_index().drop('index', axis=1)
        assert self._provided_units == item._provided_units
        assert self.miners is item.miners
        assert self.coolers is item.coolers
        return MiningProfiles(df, miners=self.miners, coolers=self.coolers, **self._provided_units)

    def implement(self, n):
        [profile.implement(n) for profile in self]

    @property
    def is_project(self):
        return np.array([p.is_project for p in self])

    @property
    def is_pool(self):
        return np.array([p.is_pool for p in self])

    @property
    def projects(self):
        return self[self.is_project]

    @property
    def pools(self):
        return self[self.is_pool]

    @property
    def miner_schedules(self):
        return np.array([profile.miner_schedule for profile in self])

    def halt(self, idxs=[]):
        if idxs:
            [p.miner.turn_off() for p in self[idxs]]

    @property
    def is_operating(self):
        return np.array([p.miner.operating for p in self])

    def operating(self):
        return self[self.is_operating]

    def summary(self):
        spacer = np.zeros(self.size)*np.nan
        summary = pd.DataFrame([
            self.n_miners,
            self.hash_rate,
            spacer,
            self.power_for_miners,
            self.power_for_cooling,
            self.power_for_miners + self.power_for_cooling,
            spacer,
            self.cost_of_miners,
            self.cost_of_cooling,
            self.build_cost,
            self.capital_cost,
            spacer,
            self.footprint,
            self.annual_property_taxes,
            ], index=[
                'Number of Miners', 'Peak Hash Rate', 
                'Power',
                'Power for Miners', 'Power for Cooling', 'Total Power',
                'Capital',
                'Cost of Miners', 'Cost of Cooling ', 'Cost of Building', 'Total Capital Cost', 
                'Other',
                'Footprint', 'Property Taxes',
            ],
            columns=self.names
        )
        return summary
