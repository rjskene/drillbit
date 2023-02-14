import pytest
import warnings
import pickle
import numpy as np
import pandas as pd

from drillbit.__new_units__ import AbstractBaseUnit, HashRate, Time

class TestUnits:
    @pytest.fixture
    def hr1(self):
        return HashRate(125, 'TH/s')

    @pytest.fixture
    def hr2(self):
        return HashRate(125, 'TH/s')

    @pytest.fixture
    def hr3(self):
       return HashRate(250, 'PH/s')

    @pytest.fixture
    def hr4(self):
        return HashRate(1, 'kH/s')

    @pytest.fixture
    def secs1(self):
        return Time(seconds=1)

    @pytest.fixture
    def secs2(self):
        return Time(minutes=10)

    def test_if_abstract_inheritance_raise_not_implemented_error(self):
        with pytest.raises(NotImplementedError):
            class TestUnit(AbstractBaseUnit):
                pass

            test = TestUnit(1, 'TH/s')

    def test_instantiation_of_hash_rate(self, hr1, hr2, hr3):
        assert isinstance(hr1, HashRate)
        assert isinstance(hr2, HashRate)
        assert isinstance(hr3, HashRate)
    
        assert repr(hr1) == '125.0 TH/s'
        assert repr(hr2) == '125.0 TH/s'
        assert repr(hr3) == '250.0 PH/s'

    def test_hash_rate_can_only_accept_correct_units(self):
        with pytest.raises(ValueError):
            HashRate(1, 'TH')
        
        with pytest.raises(ValueError):
            HashRate(1, 'TH/s/s')

    def test_addition_of_hash_rates(self, hr1, hr2, hr3, hr4):
        assert hr1 + hr2 == HashRate(250, 'TH/s')
        assert hr1 + hr3 == HashRate(250125, 'TH/s')
        assert hr1 + hr2 + hr3 == HashRate(250250, 'TH/s')

        assert hr4 + 1 == HashRate(1001, 'H/s')
        assert hr4 + 1.0 == HashRate(1001, 'H/s')
        assert 1 + hr4 == HashRate(1001, 'H/s')
        assert 1.0 + hr4 == HashRate(1001, 'H/s')

    def test_subtraction_of_hash_rates(self, hr1, hr2, hr3, hr4):
        assert hr1 - hr2 == HashRate(0, 'TH/s')
        assert hr1 - hr3 == HashRate(-249875, 'TH/s')
        assert hr1 - hr2 - hr3 == HashRate(-250000, 'TH/s')

        assert hr4 - 1 == HashRate(999, 'H/s')
        assert hr4 - 1.0 == HashRate(999, 'H/s')
        assert 1 - hr4 == HashRate(-999, 'H/s'), f'{1 - hr4}'
        assert 1.0 - hr4 == HashRate(-999, 'H/s')

    def test_multiplication_of_hash_rates_with_unitless_values(self, hr1, hr2, hr3, hr4):
        assert hr1 * 2 == HashRate(250, 'TH/s')
        assert hr1 * 2.0 == HashRate(250, 'TH/s')
        assert 2 * hr1 == HashRate(250, 'TH/s')
        assert 2.0 * hr1 == HashRate(250, 'TH/s')

    def test_division_of_hash_rates_with_unitless_values(self, hr1, hr2, hr3, hr4):
        assert hr1 / 2 == HashRate(62.5, 'TH/s')
        assert hr1 / 2.0 == HashRate(62.5, 'TH/s')
        assert 2 / hr1 == HashRate(1.6e-14, 'H/s'), f'{2 / hr1}'
        assert 2.0 / hr1 == HashRate(1.6e-14, 'H/s')

    def test_instatiation_of_time_object(self):
        pass