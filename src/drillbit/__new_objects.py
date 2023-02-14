from dataclasses import dataclass
import numpy as np
import pandas as pd

import finstat as fs

from .mining import MiningMixin, consumption_in_Wh
from .__new_units__ import Power, Energy, HashRate, HashPrice, PowerPrice, \
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

@dataclass
class Rig:
    """Object for managing mining rig properties"""
    make: str
    model: str
    generation: str
    manufacturer: str
    price: float
    power: Power
    hash_rate: HashRate
    buffer: float=0
    quantity: int=0

    def __post_init__(self):
        if not isinstance(self.power, Power):
            self.power = Power(self.power)

        if not isinstance(self.hash_rate, HashRate):
            self.hash_rate = HashRate(self.hash_rate)

    def __repr__(self):
        return (
            f'Rig(name={self.make}{self.model}{self.generation},'
            f'price={self.price},'  
            f'power={self.power},'
            f'hash_rate={self.hash_rate},' 
            f'efficiency={self.efficiency}'
        )

    def name(self):
        return f'{self.make} {self.model} {self.generation}'

    @property
    def efficiency(self):
        return self.power / self.hr

    @property
    def hr(self):
        return self.hash_rate

    @property
    def total_cost(self):
        return self.price * self.quantity

@dataclass
class Product:
    """Object for managing mining rig properties"""
    name: str
    power: Power
    pue: float
    price: float

    def __post_init__(self):
        self.quantity = None # allows inheritance of Product class

        if not isinstance(self.power, Power):
            self.power = Power(self.power)

    def __repr__(self):
        return (
            f'Product(name={self.name},'
            f'power={self.power},'
            f'pue={self.pue},'
            f'price={self.price},'  
        )

    @property
    def total_cost(self):
        return self.price * self.quantity

@dataclass
class Cooler(Product):
    """Object for managing coolers properties"""
    curve: tuple[float, float]

    def capacity_at_temp(self, temp):
        return Power(temp * self.curve[0] + self.curve[1])

class RigOperator:
    def __init__(self, rig, overclock=1):
        self.rig = rig
        self.overclock = overclock

class OverClock:
    """
    Governor for overclocking a Rig    
    """
    a = 2.6696 # units of 1 / TH/s
    b = 23.33  # units of W / TH/s
    BASERIG = Rig('Antminer', 'S19', 'Base', 'Bitmain', price=7000, power=Power(3250), hash_rate=HashRate(96, 'TH/s'))
    
    def __init__(self, factor=0, rig=None, max_power=Power(6250), func=None):
        self._power_factor = factor
        self._rig = rig
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
    def rig(self):
        return self.BASERIG if self._rig is None else self._rig
    
    @property
    def multiplier(self):
        BASERIG = object.__getattribute__(self, 'BASERIG')
        rig = object.__getattribute__(self, 'rig')
        return BASERIG.specs.efficiency / rig.specs.efficiency
    
    @property
    def power_factor(self):
        return self._power_factor
    
    def set_power_factor(self, factor):
        self._power_factor = factor

    @property
    def interval(self):
        return self.max_power - self.rig.specs.power
        
    def power_by_factor(self, n=None):
        if n is None:
            n = self._power_factor
        return Power((n * self.interval) + self.rig.specs.power)
        
    def curve(self, x=None):
        if x is None:
            x = np.linspace(self.rig.specs.power, self.max_power, 100)
        return x, self.func(x)
    
    def eff_by_power(self, x=None):
        if x is None:
            x = self.rig.specs.power
        return Efficiency(self.func(x))
    
    def eff_by_factor(self, n=None):
        return Efficiency(self.func(self.power_by_factor(n)))
    
    def hash_rate_by_power(self, x=None):
        if x is None:
            x = self.rig.specs.power
        return self.eff_by_factor(x).hash_rate(x)
    
    def hash_rate_by_factor(self, n=None):
        power = self.power_by_factor(n)
        return self.eff_by_factor(n).hash_rate(power)

@dataclass
class Project:
    capacity: Power
    rigs: Rig
    infrastructure: list[Product]
    target_overclocking: OverClock = 1
    energy_price: EnergyPrice = 0
    target_ambient_temp: float = 95

    def __post_init__(self):
        if isinstance(self.rigs, list):
            self.rigs = self.rigs[0] # For now, override rigs to be a single rig

        if not isinstance(self.capacity, Power):
            self.capacity = Power(self.capacity)

        if not isinstance(self.energy_price, EnergyPrice):
            self.energy_price = EnergyPrice(self.energy_price)

    def _get_scaler(self):
        return ProjectScaler(self)

    def scale(self):
        scaler = self._get_scaler()
        scaler.assign_quantities()
        return self

class ProjectScaler:
    def __init__(self, project):
        self.project = project
        
    def compute_power_per_rig(self):
        return self.project.rigs.power * (1 + self.project.rigs.buffer) * self.project.target_overclocking
    
    def project_pue(self):
        return np.prod(np.array([p.pue for p in self.project.infrastructure]))
    
    def total_power_per_rig(self):
        return self.compute_power_per_rig() * self.project_pue()
    
    def compute_power(self):
        return self.compute_power_per_rig() * self.rig_quantity()
    
    def rig_quantity(self):
        return self.project.capacity / self.total_power_per_rig()
    
    def infrastructure_quantity(self, product):
        if hasattr(product, 'curve'):
            power_per_unit = product.capacity_at_temp(self.project.target_ambient_temp)
        else:
            power_per_unit = product.power

        return self.compute_power() / power_per_unit

    def assign_rig_quantity(self):
        self.project.rigs.quantity = self.rig_quantity()

    def assign_infrastructure_quantity(self):
        for product in self.project.infrastructure:
            product.quantity = self.infrastructure_quantity(product)

    def assign_quantities(self):
        self.assign_rig_quantity()
        self.assign_infrastructure_quantity()
