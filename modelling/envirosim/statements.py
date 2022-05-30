
import numpy as np
import pandas as pd

import finstat as fs

from bitcoin.statements.funcs import *
from bitcoin.statements import init_enviro

class MineTemplate:
    def __new__(self, mine, miner_energy, hashes, **kwargs):
        stat = fs.FinancialStatement(name=mine.name, short_name=mine.short_name, **kwargs)

        stat.add_statement(name='Environment', short_name='env')
        stat.add_statement(name='Income Statement')
        
        stat.env.add_account(mine.miner_schedule, name='Number of Miners', short_name='n_miners')

        stat.env.add_account(miner_energy, name='Energy - Miner', short_name='miner_energy')
        stat.env.add_account(cooling_energy(stat.miner_energy, mine.cooling.pue), name='Energy - Cooling', short_name='cool_energy')
        stat.env.add_account(fs.arr.add(stat.cool_energy, stat.miner_energy), name='Energy')

        stat.env.add_account(hashes, name='Hashes')
        stat.env.add_account(hashes_to_hash_rate(hashes, periods=stat.periods), name='Hash Rate')

        return stat

    @staticmethod
    def finalize(stat, env, mine):
        stat.env.add_account(win_percentage(stat.hashes, env.difficulty), name='Win Share', short_name='win_share')

        stat.env.add_account(fs.arr.multiply(stat.win_share, env.reward), name='BTC Reward', short_name='btc_reward')
        stat.env.add_account(fs.arr.multiply(stat.win_share, env.fees), name='Transaction Fees', short_name='traxn_fees')
        stat.env.add_account(fs.arr.add(stat.btc_reward, stat.traxn_fees), name='BTC Mined', short_name='btc_mined')

        stat.istat.add_account(fs.arr.multiply(stat.btc_reward, env.btc_price), name='Revenue - Reward', short_name='reward_rev')
        stat.istat.add_account(fs.arr.multiply(stat.traxn_fees, env.btc_price), name='Revenue - Fees', short_name='fee_rev')
        stat.istat.add_account(fs.arr.add(stat.reward_rev, stat.fee_rev), name='Revenue', short_name='rev')

        stat.istat.add_account(fs.arr.multiply(stat.energy, mine.energy_cost), name='Energy Expenses', short_name='energy_exp')
        stat.istat.add_account(fs.arr.add(stat.rev, -stat.energy_exp), name='Gross Profit')
        stat.istat.add_account(fs.arr.divide(stat.gp, stat.rev), name='Gross Margin')

class MineStats(np.ndarray):
    def __new__(cls, mines, miner_energy, hashes, periods, **kwargs):
        minerstats = [MineTemplate(m, e, h, periods=periods, no_model=True, **kwargs) for m, e, h in zip(mines, miner_energy, hashes)]
        assert np.all(minerstats[0].lineitems.short_names == [s.lineitems.short_names for s in minerstats])
        
        obj = np.asarray([o for o in minerstats], dtype='object').view(cls)
        obj.mines = mines
        obj.periods = periods
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.mines = getattr(obj, 'mines', None)
        self.periods = getattr(obj, 'periods', None)

    def __getattribute__(self, name):
        names = object.__getattribute__(self, 'names')
        short_names = object.__getattribute__(self, 'short_names')

        if name == '__array_finalize__':
            return super().__getattribute__(name)
        else:
            if name in names:
                return self[name == names][0]
            elif name in short_names:
                return self[name == short_names][0]
            else:
                return super().__getattribute__(name)

    @property
    def names(self):
        return np.array([p.name for p in self])

    @property
    def short_names(self):
        return np.array([p.short_name for p in self])

    def items(self):
        for mine, stat in zip(self.mines, self):
            yield mine, stat

    @property
    def is_project(self):
        return np.array([p.is_project for p in self.mines])

    @property
    def is_pool(self):
        return np.array([p.is_pool for p in self.mines])

    @property
    def pools(self):
        pools = self[self.is_pool]
        pools.mines = self.mines[self.is_pool]
        return pools

    @property
    def projects(self):
        projects = self[self.is_project]
        projects.mines = self.mines[self.is_project]
        return projects

    def by_lineitem(self, short_name, how='list'):
        items = [getattr(mstat, short_name) for mstat in self]

        if how == 'list':
            return items
        elif how == 'frame':
            return pd.concat(items, keys=[m.name for m in self.mines], axis=1).T
        else:
            raise

    def finalize(self, env, network_hashes):
        env.add_account(network_hashes, name='Network Hashes', short_name='net_hashes')
        env.add_account(expected_difficulty(env.net_hashes), name='Difficulty')
        env.add_account(hashes_to_hash_rate(env.net_hashes), name='Network Hash Rate', short_name='net_hr')

        for mine, minestat in self.items():
            MineTemplate.finalize(minestat, env, mine)
