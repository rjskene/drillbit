import warnings
from abc import ABC, abstractmethod

import datetime as dt
import math
import numpy as np
import pandas as pd

__all__ = [
    'Energy', 'Power', 'Hashes', 'HashRate', 'Efficiency', 'Density', 'Area',
    'EnergyPrice', 'PowerPrice', 'HashPrice', 'PropertyPrice',
]

class MagnitudeTable:
    MAGNITUDES = {
        'base': {'order': 0, 'abbr': ''},
        'kilo': {'order': 3, 'abbr': 'k'},
        'mega': {'order': 6, 'abbr': 'M'},
        'giga': {'order': 9, 'abbr': 'G'},
        'tera': {'order': 12, 'abbr': 'T'},
        'peta': {'order': 15, 'abbr': 'P'},
        'exa': {'order': 18, 'abbr': 'E'},
        'zetta': {'order': 21, 'abbr': 'Z'},
        'yotta': {'order': 24, 'abbr': 'Y'},
        'ronna': {'order': 27, 'abbr': 'R'},
    }
    
    def __new__(cls, units, inverse=False):
        return cls.make(units, inverse)
    
    @classmethod
    def make(cls, units, inverse):
        df = pd.DataFrame(cls.MAGNITUDES).T

        units = units if '{}' in units else '{}' + units
        df.loc[:, 'units'] = df.abbr.apply(lambda val: units.format(val))

        df = df.drop(columns=['abbr']) \
            .reset_index() \
            .rename(columns={'index': 'abbr'})

        if inverse:
            df.order *= -1

        df.inverse = inverse

        return df

class AbstractBaseUnit(ABC, float):
    # @classmethod
    # @property
    # @abstractmethod
    # def MAGNITUDES(self):
    #     raise NotImplementedError

    def __new__(cls, value, units=None):
        if units is not None:
            if not (cls.MAGNITUDES.units == units).any():
                raise ValueError((
                    'Provided units not supported. '
                    'Please provide one of the following: '
                    f"{', '.join(cls.MAGNITUDES.units)}"
                ))
                
            order = cls.MAGNITUDES.set_index('units').loc[units].order
            value = cls.return_value_value(value, order)

        return float.__new__(cls, value)
        
    def __init__(self, value, abbr=None, **kwargs):
        float.__init__(value)
    
    def __repr__(self):
        """
        Override the default __repr__ method to return the value of the
        object in its most concise units.

        `value` and `magnitude` objects must be fetched via `object.__getattribute__`
        to avoid infinite recursion.

        Formula: the unscaled value value is scaled to the nearest concise magnitude
        and the units are appended to the scaled value.

        Returns
        -------
        str;    The value of the object in its most concise units.
        """
        value = object.__getattribute__(self, 'value')
        magnitude = object.__getattribute__(self, 'magnitude')   
        return f'{value / (10**magnitude.order)} {magnitude.units}'

    def __add__(self, value):
        res = super().__add__(value)
        return self.__class__(res, **self.__dict__)

    def __radd__(self, value):
        res = super().__radd__(value)
        return self.__class__(res, **self.__dict__)

    def __sub__(self, value):
        res = super().__sub__(value)
        return self.__class__(res, **self.__dict__)
    
    def __rsub__(self, value):
        res = super().__rsub__(value)
        return self.__class__(res, **self.__dict__)

    def __mul__(self, value):
        res = super().__mul__(value)
        return self.__class__(res, **self.__dict__)

    def __rmul__(self, value):
        res = super().__rmul__(value)
        return self.__class__(res, **self.__dict__)

    def __truediv__(self, value):
        res = super().__truediv__(value)

        if isinstance(value, self.__class__):
            return res
        else:
            return self.__class__(res, **self.__dict__)

    def __rtruediv__(self, value):
        res = super().__rtruediv__(value)

        if isinstance(value, self.__class__):
            return res
        else:
            return self.__class__(res, **self.__dict__)

    @property
    def _value_order(self):
        """
        Finds the base 10 order of the value.

        If value is 0, returns 0. This will ensure that index 0 is selected
        from the MAGNITUDES table.

        If value is negative, returns the order of the absolute value.
        """
        if self.value:
            if self.value < 0:
                return round(math.log(-self, 10),0)
            else:
                return round(math.log(self, 10))
        else:
            return 0
    
    @property
    def _magnitude_index(self):
        """
        Returns index of row in magnitude table that is closest to the
        value's order. 

        If value is less than 0, returns 0, so any magnitudes less than the base unit
        will be represented in the base unit.
        """
        index = math.floor(self._value_order / 3)
        
        if self.MAGNITUDES.inverse and self._value_order <= 0:
            return -index
        elif self._value_order >= 0:
            return index
        else:
            return 0

    @property
    def magnitude(self):
        return self.MAGNITUDES.iloc[self._magnitude_index]
        
    @classmethod
    def return_value_value(cls, value, order):
        return value * (10**order)

    @property
    def value(self):
        return self.__float__()

class MultiUnitMixin:
    def __new__(cls, value, units=None, **kwargs):
        cls.MAGNITUDES = cls._get_magnitudes_(cls, units)
        return super().__new__(cls, value, units=units, **kwargs)

    @staticmethod
    def _get_magnitudes_(cls, units=None):
        if units is None:
            return cls.DEFAULT_TABLE
        else:
            MAGNITUDE_TABLES = object.__getattribute__(cls, 'MAGNITUDE_TABLES')

            for table in MAGNITUDE_TABLES:
                if (table.units == units).any():
                    return table

        raise ValueError((
            'Provided units not supported. '
            'Please provide one of the following: '
            f"{', '.join(pd.concat(MAGNITUDE_TABLES).units)}"
        ))

class PowerConversions:
    def in_joules(self):
        warnings.warn('Watts and Joules per Second have same magnitude')
        if (self.MAGNITUDES.units == 'J/s').any():
            warnings.warn('Units are already Joules per Second')            
        else:
            self.MAGNITUDES = self.MAGNITUDES_J

        return self

    def in_watts(self):
        warnings.warn('Watts and Joules per Second have same magnitude')
        if (self.MAGNITUDES.units == 'W').any():
            warnings.warn('Units are already in Watts')            
        else:
            self.MAGNITUDES = self.MAGNITUDES_W

        return self

class EnergyConversions:
    def in_joules(self):
        if (self.MAGNITUDES.units == 'J').any():
            warnings.warn('Units are already Joules')            
        else:
            return self.__class__(self.value * 60 * 60, units='J')    

    def in_watts(self):
        if (self.MAGNITUDES.units == 'Wh').any():
            warnings.warn('Units are already Watt-hours')            
        else:
            return self.__class__(self.value / 60 / 60, units='Wh')    

class Time(float):
    def __new__(cls, *args, **kwargs):
        cls._timedelta = dt.timedelta(*args, **kwargs)

        return float.__new__(cls, cls._timedelta.seconds)

    def __repr__(self):
        timedelta = object.__getattribute__(self, '_timedelta')
        return f'{timedelta.seconds:,.2f} s'

    def __add__(self, value):
        res = super().__add__(value)
        return self.__class__(seconds=res, **self.__dict__)

    def __radd__(self, value):
        res = super().__radd__(value)
        return self.__class__(seconds=res, **self.__dict__)

    def __sub__(self, value):
        res = super().__sub__(value)
        return self.__class__(seconds=res, **self.__dict__)
    
    def __rsub__(self, value):
        res = super().__rsub__(value)
        return self.__class__(seconds=res, **self.__dict__)

    def __mul__(self, value):
        res = super().__mul__(value)
        return self.__class__(seconds=res, **self.__dict__)

    def __rmul__(self, value):
        res = super().__rmul__(value)
        return self.__class__(seconds=res, **self.__dict__)

    def __truediv__(self, value):
        res = super().__truediv__(value)
        return self.__class__(seconds=res, **self.__dict__)

    def __rtruediv__(self, value):
        res = super().__rtruediv__(value)
        return self.__class__(seconds=res, **self.__dict__)

    @property
    def value(self):
        return self.__float__()

class Hashes(AbstractBaseUnit):
    MAGNITUDES = MagnitudeTable('H')
    
    def __truediv__(self, value):
        res = super().__truediv__(value)
        if isinstance(value, Time):
            res = HashRate(res)
        elif isinstance(value, Energy):
            res = Efficiency(res)

        return res

    def rate_per_block(self):
        return HashRate(self / 600)

class HashRate(AbstractBaseUnit):
    MAGNITUDES = MagnitudeTable('H/s')

    def __mul__(self, value):
        res = super().__mul__(value)

        if isinstance(value, Efficiency):
            return Power(res, 'W')
        elif isinstance(value, Time):
            res = Hashes(res)

        return res

    def hashes_per_block(self):
        return Hashes(self *  60 * 10)

    def hashes_per_day(self):
        return Hashes(self * 60 * 10 * 6 * 24)

    def hashes_per_month(self, days=30):
        return Hashes(self * 60 * 10 * 6 * 24 * days)

    def hashes_per_year(self, days=365):
        return Hashes(self * 60 * 10 * 6 * 24 * days)

class Power(MultiUnitMixin, AbstractBaseUnit, PowerConversions):
    MAGNITUDES_W = MagnitudeTable('W')
    MAGNITUDES_J = MagnitudeTable('J/s')
    MAGNITUDE_TABLES = [MAGNITUDES_W, MAGNITUDES_J]
    DEFAULT_TABLE = MAGNITUDES_W

    def __truediv__(self, value):
        res = super().__truediv__(value)

        if isinstance(value, HashRate):
            res = Efficiency(res)
        elif isinstance(value, Efficiency):
            res = HashRate(res)

        return res

    def consumption(self, **duration):
        from .mining import consumption_in_Wh
        return consumption_in_Wh(self, **duration)

    def consumption_per_block(self):
        return self.consumption(minutes=10)

class Energy(MultiUnitMixin, AbstractBaseUnit, EnergyConversions):
    MAGNITUDES_W = MagnitudeTable('Wh')
    MAGNITUDES_J = MagnitudeTable('J')
    MAGNITUDE_TABLES = [MAGNITUDES_W, MAGNITUDES_J]
    DEFAULT_TABLE = MAGNITUDES_W

class Efficiency(MultiUnitMixin, AbstractBaseUnit, PowerConversions):
    MAGNITUDES_W = MagnitudeTable('W / {}H/s', inverse=True)
    MAGNITUDES_J = MagnitudeTable('J / {}H', inverse=True)
    MAGNITUDE_TABLES = [MAGNITUDES_W, MAGNITUDES_J]
    DEFAULT_TABLE = MAGNITUDES_W    

    def __mul__(self, value):
        res = super().__truediv__(value)

        if isinstance(value, Hashes):
            return Energy(res, units='J').in_watts()
        elif isinstance(value, HashRate):
            return Power(res, units='W')

        return res

    def hash_rate(self, power):
        return HashRate(power / self)

class Density(AbstractBaseUnit):
    MAGNITUDES = MagnitudeTable('{}W / sf')

class Area(AbstractBaseUnit):
    MAGNITUDES = MagnitudeTable('sf', '{}sf')

class EnergyPrice(MultiUnitMixin, AbstractBaseUnit, EnergyConversions):
    MAGNITUDES_W = MagnitudeTable('$ / {}Wh', inverse=True)
    MAGNITUDES_J = MagnitudeTable('$ / {}J', inverse=True)
    MAGNITUDE_TABLES = [MAGNITUDES_W, MAGNITUDES_J]
    DEFAULT_TABLE = MAGNITUDES_W

class PowerPrice(MultiUnitMixin, AbstractBaseUnit, PowerConversions):
    MAGNITUDES_W = MagnitudeTable('$ / {}W', inverse=True)
    MAGNITUDES_J = MagnitudeTable('$ / {}J/s', inverse=True)
    MAGNITUDE_TABLES = [MAGNITUDES_W, MAGNITUDES_J]
    DEFAULT_TABLE = MAGNITUDES_W

    def cost_per_block(self):
        return EnergyPrice(self * 60 * 10)

class HashPrice(AbstractBaseUnit):
    MAGNITUDES = MagnitudeTable('$ / {}H', inverse=True)

class PropertyPrice(AbstractBaseUnit):
    MAGNITUDES = MagnitudeTable('$ / {}sf', inverse=True)

class UnitsArray(np.ndarray):
    """
    A numpy array of Units objects
    """

    def __new__(cls, elements):
        obj = np.asarray([a for a in elements], dtype='object').view(cls)
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return

    def __repr__(self):
        joined = ', '.join([s.__repr__() for s in self])
        return f'UnitsArray([{joined}])'

    @property
    def values(self):
        return np.array([s.value for s in self])

    def in_joules(self):
        return UnitsArray([s.in_joules() for s in self])

    def in_watts(self):
        return UnitsArray([s.in_watts() for s in self])
