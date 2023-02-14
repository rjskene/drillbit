import warnings
import numpy as np
import pandas as pd

__all__ = [
    'Energy', 'Power', 'Hashes', 'HashRate', 'Efficiency', 'Density', 'Area',
    'EnergyPrice', 'PowerPrice', 'HashPrice', 'PropertyPrice',
]


class MagnitudeTable:
    MAGNITUDES = {
        'base': {'magnitude': 0, 'abbr': ''},
        'kilo': {'magnitude': 3, 'abbr': 'k'},
        'mega': {'magnitude': 6, 'abbr': 'M'},
        'giga': {'magnitude': 9, 'abbr': 'G'},
        'tera': {'magnitude': 12, 'abbr': 'T'},
        'peta': {'magnitude': 15, 'abbr': 'P'},
        'exa': {'magnitude': 18, 'abbr': 'E'},
        'zetta': {'magnitude': 21, 'abbr': 'Z'},
        'yotta': {'magnitude': 24, 'abbr': 'Y'},
        'ronna': {'magnitude': 27, 'abbr': 'R'},
    }
    
    def __new__(cls, abbr, units, inverse=False):
        return cls.make(abbr, units, inverse)
    
    @classmethod
    def make(cls, abbr, units, inverse):
        df = pd.DataFrame(cls.MAGNITUDES).T

        magkey = df.copy()
        if inverse:
            magkey.magnitude *= -1

        magkey.index = df.abbr + abbr
        magkey = magkey.rename(columns={'abbr': 'units'})

        units = units if '{}' in units else '{}' + units
        magkey.units = magkey.units.apply(lambda val: units.format(val))

        return magkey

class BaseUnits(float):
    _round = 2

    def __new__(cls, value, abbr=None, magkey=None, **kwargs):
        if magkey is None:
            magkey = cls.MAGKEY
        
        if value == 0:
            abbr = magkey.index[0]

        if abbr is not None:
            mag = magkey.loc[abbr].magnitude
            value = cls.unscaled(value, mag)

        try:
            iterator = iter(value)
        except TypeError:
            return float.__new__(cls, value)
        else:
            return UnitsArray([cls(v, abbr=None) for v in value])
        
    def __init__(self, value, abbr=None, **kwargs):
        float.__init__(value)
        
        if value == 0:
            abbr = self.MAGKEY.index[0]

        if abbr is None:
            self.abbr = self.magrow.name
        else:
            self.abbr = abbr
        
    def __getattribute__(self, name):
        magkey = object.__getattribute__(self, 'magkey')

        if name in magkey.index:
            mag = magkey.loc[name].magnitude
            scaled_func = object.__getattribute__(self, 'scaled')
            constructor = object.__getattribute__(self, '__class__')

            return constructor(scaled_func(mag), abbr=name)
        else:
            return super().__getattribute__(name)

    def __repr__(self):
        mag, units = self.magrow
        return f'{self.scaled(mag):,.{self._round}f}'.rstrip('0').rstrip('.') + f' {units}'
        
    def as_scale(self):
        return self.scaled(self.magnitude)

    def round(self, n):
        self._round = n
        return self

    def add(self, value):
        res = super().__add__(value)
        res /= 10**self.magnitude
        return self.__class__(res, **self.__dict__)
    
    def subtract(self, value):
        res = super().__sub__(value)
        res /= 10**self.magnitude
        return self.__class__(res, **self.__dict__)

    def __mul__(self, value):
        res = super().__mul__(value)
        res /= 10**self.magnitude
        return self.__class__(res, **self.__dict__)

    def __rmul__(self, value):
        res = super().__mul__(value)
        res /= 10**self.magnitude
        return self.__class__(res, **self.__dict__)

    def multiply(self, value):
        res = super().__mul__(value)
        res /= 10**self.magnitude
        return self.__class__(res, **self.__dict__)

    def divide(self, value):
        res = super().__truediv__(value)
        res /= 10**self.magnitude
        return self.__class__(res, **self.__dict__)

    @property
    def MAGKEY(self):
        raise NotImplementedError

    @property
    def magkey(self):
        return object.__getattribute__(self, 'MAGKEY')
    
    @property
    def value(self):
        return float(self)
    
    @property
    def magmask(self):
        raise NotImplementedError

    @property
    def magnitude(self):
        return self.magrow.magnitude

    @property
    def units(self):
        return self.magrow.units

    @property
    def magrow(self):
        if not hasattr(self, 'abbr') or self.abbr is None:
            return self.magkey[self.magmask].iloc[-1]
        else:
            return self.magkey.loc[self.abbr]
    
    def scaled(self, mag=None):
        mag = self.magnitude if mag is None else mag
        return float(self) / (10**mag)

    @classmethod
    def unscaled(cls, value, mag):
        return value * (10**mag)
    
class Units(BaseUnits):
    @property
    def magmask(self):
        return self.magkey.magnitude.between(0, np.log10(self))

class InverseUnits(BaseUnits):
    @property
    def magmask(self):
        return self.magkey.magnitude.between(np.log10(self) - 3, 0)

class MultiUnitMixin:
    def __new__(cls, value, abbr=None, **kwargs):
        cls.MAGKEY = cls._get_magkey_(cls, abbr)
        cls._magkey = cls.MAGKEY
        return super().__new__(cls, value, abbr=abbr, **kwargs)

    def __init__(self, value, abbr=None, **kwargs):
        self._magkey = MultiUnitMixin._get_magkey_(self, abbr)
        super().__init__(value, abbr, **kwargs)        
    
    @staticmethod
    def _get_magkey_(cls, abbr):
        MAGKEYS = object.__getattribute__(cls, 'MAGKEYS')
        if abbr is None:
            return MAGKEYS[0]
        else:
            for magkey in MAGKEYS:
                if abbr in magkey.index:
                    return magkey
            raise ValueError('abbr not found')
        
    @classmethod
    @property
    def MAGKEYS(cls):
        return NotImplementedError

    @property
    def magkey(self):
        return object.__getattribute__(self, '_magkey')

class EnergyConversions:
    def in_joules(self):
        if 'J' in self.magkey.index:
            warnings.warn('You are already in Joules')
            return self
        else:
            abbr = '' if self.abbr is None else self.abbr.rstrip('Wh')
            magnitude, _ = self.magrow
            return self.__class__(self.scaled(magnitude) * 60 * 60, abbr=abbr + 'J')    

    def in_watts(self):
        if 'Wh' in self.magkey.index:
            warnings.warn('You are already in Watt-Hours')
            return self
        else:
            abbr = '' if self.abbr is None else self.abbr.rstrip('J')
            magnitude, _ = self.magrow
            return self.__class__(self.scaled(magnitude) / 60 / 60, abbr=abbr + 'Wh')

class PowerConversions:
    def in_joules(self):
        warnings.warn('Watts and Joules per Second have same magnitude')
        abbr = self.abbr.rstrip('W')
        magnitude, _ = self.magrow
        return self.__class__(self.scaled(magnitude), abbr=abbr + 'J')

    def in_watts(self):
        warnings.warn('Watts and Joules per Second have same magnitude')
        abbr = self.abbr.rstrip('J')
        magnitude, _ = self.magrow
        return self.__class__(self.scaled(magnitude), abbr=abbr + 'W')

class Energy(MultiUnitMixin, Units, EnergyConversions):
    MAGKEY_A = MagnitudeTable('Wh', 'Wh')
    MAGKEY_B = MagnitudeTable('J', 'J')
    MAGKEYS = [MAGKEY_A, MAGKEY_B]
    
class Power(MultiUnitMixin, Units, PowerConversions):
    MAGKEY_A = MagnitudeTable('W', 'W')
    MAGKEY_B = MagnitudeTable('J', 'J / s')
    MAGKEYS = [MAGKEY_A, MAGKEY_B]

    def __truediv__(self, value):
        if isinstance(value, Efficiency):
            return HashRate(self.value / value)
        if isinstance(value, HashRate):
            return Efficiency(self.value / value)
        else:
            return super().__truediv__(value)

    @property
    def is_joules(self):
        return 'J' in self.magkey.index

    def consumption(self, **duration):
        from .mining import consumption_in_Wh
        return consumption_in_Wh(self, **duration)

    def consumption_per_block(self):
        return self.consumption(minutes=10)

class Hashes(Units):
    MAGKEY = MagnitudeTable('H', 'H')
    
    def __mul__(self, value):
        if isinstance(value, Efficiency):
            return value.__mul__(self)
        else:
            return super().__mul__(value)

    def rate(self):
        return HashRate(self.value / 600)

class HashRate(Units):
    MAGKEY = MagnitudeTable('H', 'H / s')

    def __mul__(self, value):
        if isinstance(value, Efficiency):
            return value.__mul__(self)
        else:
            return super().__mul__(value)

    def hashes_per_block(self):
        return Hashes(self * 60 * 10)

    def hashes_per_day(self):
        return Hashes(self * 60 * 10 * 6 * 24)

    def hashes_per_month(self, days=30):
        return Hashes(self * 60 * 10 * 6 * 24 * days)

    def hashes_per_year(self, days=365):
        return Hashes(self * 60 * 10 * 6 * 24 * days)

class Efficiency(MultiUnitMixin, InverseUnits):
    W_MAGKEY = MagnitudeTable('H_W', 'W / {}H/s', inverse=True)
    J_MAGKEY = MagnitudeTable('H_J', 'J / {}H', inverse=True)
    MAGKEYS = [W_MAGKEY, J_MAGKEY]
    
    def __mul__(self, value):
        if isinstance(value, Hashes):
            return Energy(self.value * value, abbr='J').in_watts()
        elif isinstance(value, HashRate):
            return Power(self.value * value)
        else:
            return super().__mul__(value)

    def in_joules(self):
        warnings.warn('Efficiency has same magnitude expressed in Watts or Joules')
        abbr = '' if self.abbr is None else self.abbr.rstrip('W')
        magnitude, _ = self.magrow
        return self.__class__(self.scaled(magnitude), abbr=abbr + 'J')
    
    def in_watts(self):
        warnings.warn('Efficiency has same magnitude expressed in Watts or Joules')
        abbr = self.abbr.rstrip('J')
        magnitude, _ = self.magrow
        return self.__class__(self.scaled(magnitude), abbr=abbr + 'W')

    def hash_rate(self, power):
        return HashRate(power / self)

class Density(Units):
    MAGKEY = MagnitudeTable('W', '{}W / sf')

class Area(Units):
    MAGKEY = MagnitudeTable('sf', '{}sf')

class EnergyPrice(MultiUnitMixin, InverseUnits, EnergyConversions):
    MAGKEY_A = MagnitudeTable('Wh', '$ / {}Wh', inverse=True)
    MAGKEY_B = MagnitudeTable('J', '$ / {}J', inverse=True)
    MAGKEYS = [MAGKEY_A, MAGKEY_B]

class PowerPrice(MultiUnitMixin, InverseUnits, PowerConversions):
    MAGKEY_A = MagnitudeTable('W', '$ / {}W', inverse=True)
    MAGKEY_B = MagnitudeTable('J', '$ / {}J/s', inverse=True)
    MAGKEYS = [MAGKEY_A, MAGKEY_B]

    def cost_per_block(self):
        return EnergyPrice(self * 60 * 10)

class HashPrice(InverseUnits):
    MAGKEY = MagnitudeTable('H', '$ / {}H', inverse=True)

class PropertyPrice(InverseUnits):
    MAGKEY = MagnitudeTable('sf', '$ / {}sf', inverse=True)

class UnitsArray(np.ndarray):
    """
    Container for LineItem objects. Standard numpy methods and attributes are available.

    Includes several helper methods for manipulating sets of LineItem objects used in 
    MetricFunction and FinancialStatment.

    Issues arise from assigning LineItem objects directly to `obj`, so the str representation is instead.
    The underlying items manipulated are still the LineItem objects, given changes to __getitem__

    Parameters:
        elements: iterable of LineItem objects
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
