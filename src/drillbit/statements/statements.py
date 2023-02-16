import numpy as np
import pandas as pd
import numpy_financial as npf

from .funcs import total_energy, expected_difficulty, hashes_to_hash_rate, \
    hash_rate_to_hashes, win_percentage, miner_amort, amort_sched

import finstat as fs

def init_environment(block_schedule, price, fees, hash_rate):
    stat = fs.FinancialStatement(name='BTC World', periods=block_schedule.period)

    stat.add_account(block_schedule.block_id.astype('int'), name='Block ID', short_name='block_id')
    stat.add_account(block_schedule.reward, name='Block Reward', short_name='reward')
    stat.add_account(price, name='BTC Price', short_name='btc_price')
    stat.add_account(fees, name='Transaction Fees', short_name='fees')
    stat.add_account(hash_rate, name='Network Hash Rate', short_name='net_hr')

    stat.add_account(fs.arr.multiply(stat.reward, stat.btc_price), name='Market Rewards', short_name='mkt_rewards')
    stat.add_account(fs.arr.multiply(stat.fees, stat.btc_price), name='Market Fees', short_name='mkt_fees')
    stat.add_account(fs.arr.add(stat.mkt_rewards, stat.mkt_fees), name='Market Revenue', short_name='mkt_rev')

    stat.add_account(hash_rate_to_hashes(stat.net_hr), name='Network Hashes', short_name='net_hashes')
    stat.add_account(expected_difficulty(stat.net_hashes), name='Difficulty')
    
    return stat

class ProjectTemplate:
    def __new__(self, mine, tax_rate=0, **kwargs):
        stat = fs.FinancialStatement(name=mine.name, short_name=mine.short_name, **kwargs)
        stat.add_factor('tax_rate', tax_rate)

        stat.add_statement(name='Environment', short_name='env')
        stat.add_statement(name='Income Statement')
        
        stat.env.add_account(mine.miner_schedule, name='Number of Miners', short_name='n_miners')
        stat.env.add_account(mine.miner.consumption_variance(stat.periods.size), name='Power Variance', short_name='pow_var', hide=True)
        stat.env.add_account(
            fs.arr.multiply(stat.n_miners, mine.miner.power.consumption_per_block().in_joules(), stat.pow_var), 
            name='Energy (J) - Miner', 
            short_name='miner_energy_in_joules',
            hide=True
        )

        stat.env.add_account(fs.arr.multiply(stat.n_miners, mine.miner.power.consumption_per_block(), stat.pow_var), name='Energy - Miner', short_name='miner_energy')
        stat.env.add_account(total_energy(stat.miner_energy, mine.cooling.pue), name='Energy - Cooling', short_name='cool_energy')
        stat.env.add_account(fs.arr.add(stat.cool_energy, stat.miner_energy), name='Energy')

        stat.env.add_account(fs.arr.multiply(mine.miner.hr, stat.n_miners), name='Hash Rate')
        stat.env.add_account(fs.arr.multiply(mine.miner.hr.hashes_per_block(), stat.n_miners), name='Hashes')

        return stat

    @staticmethod
    def finalize(stat, mine, env):
        stat.env.add_account(win_percentage(stat.hashes, env.difficulty), name='Win %', short_name='win_per')

        stat.env.add_account(fs.arr.multiply(stat.win_per, env.reward), name='BTC Reward', short_name='btc_reward')
        stat.env.add_account(fs.arr.multiply(stat.win_per, env.fees), name='Transaction Fees', short_name='traxn_fees')
        stat.env.add_account(fs.arr.multiply(stat.btc_reward, mine.pool_fee), name='Pool Fees (\u0243)', short_name='pool_fees_in_btc')
        stat.env.add_account(fs.arr.add(stat.btc_reward, stat.traxn_fees, -stat.pool_fees_in_btc), name='BTC Mined', short_name='btc_mined')

        stat.istat.add_account(fs.arr.multiply(stat.btc_reward, env.btc_price), name='Revenue - Reward', short_name='reward_rev')
        stat.istat.add_account(fs.arr.multiply(stat.traxn_fees, env.btc_price), name='Revenue - Fees', short_name='fee_rev')
        stat.istat.add_account(fs.arr.add(stat.fee_rev, stat.reward_rev), name='Gross Revenue', short_name='gross_rev')

        stat.istat.add_account(fs.arr.multiply(stat.pool_fees_in_btc, env.btc_price), name='Pool Fees', short_name='pool_fees')
        stat.istat.add_account(fs.arr.add(stat.gross_rev, -stat.pool_fees), name='Net Revenue', short_name='net_rev')
        stat.istat.add_account(fs.arr.multiply(stat.btc_mined, env.btc_price), name='Test Net Revenue', short_name='test_net_rev', hide=True)

        stat.istat.add_account(fs.arr.multiply(stat.energy, mine.energy_cost), name='Energy Expenses', short_name='energy_exp')
        stat.istat.add_account(fs.arr.add(stat.net_rev, -stat.energy_exp), name='Gross Profit')
        stat.istat.add_account(fs.arr.divide(stat.gp, stat.net_rev), name='Gross Margin', hide=True)

        stat.istat.add_account(fs.arr.multiply(stat.energy, mine.opex_cost.cost_per_block()), name='Operations', short_name='ops')
        stat.istat.add_account(fs.arr.repeat(mine.property_taxes_per_block, stat.periods.size, mine.implement.start_in_blocks(), periods=stat.periods), name='Property Taxes', short_name='prop_tax')

        stat.istat.add_account(fs.arr.add(stat.gp, -stat.ops, -stat.prop_tax), name='EBITDA')

        stat.istat.add_account(miner_amort(mine, periods=stat.periods), name='Miner Amortization', short_name='miner_amort')
        stat.istat.add_account(amort_sched(mine.build_cost, 1, mine.property_amort, mine.implement.start_in_blocks(), periods=stat.periods), name='Building Amortization', short_name='build_amort')
        stat.istat.add_account(amort_sched(mine.cost_of_cooling, 1, mine.cooling.amort, mine.implement.start_in_blocks(), periods=stat.periods), name='Cooling Amortization', short_name='cool_amort')

        stat.istat.add_account(fs.arr.add(stat.miner_amort, stat.build_amort, stat.cool_amort), name='Depreciation for Taxes', short_name='tax_depn')

        stat.istat.add_account(fs.arr.add(stat.ebitda, -stat.tax_depn), name='EBIT')
        stat.istat.add_account(fs.arr.multiply(stat.ebit, stat.tax_rate), name='Taxes')

        stat.istat.add_account(fs.arr.add(stat.ebit, -stat.taxes), name='Profit, if sold', short_name='profit_sold')
        stat.istat.add_account(fs.arr.add(stat.ebitda, -stat.taxes), name='Operating Cash Flow, if sold', short_name='op_flow_sold')

        stat.istat.add_account(fs.arr.add(stat.energy_exp, stat.ops, stat.prop_tax, stat.taxes), name='Cash Expenses', short_name='cash_exp', hide=True)
        stat.istat.add_account(fs.arr.divide(stat.cash_exp, env.btc_price), name='BTC Converted for Expenses', short_name='converted')
        
        stat.istat.add_account(fs.arr.add(stat.btc_mined, -stat.converted), name='BTC Earned', short_name='btc_earned')
        stat.istat.add_account(fs.arr.cumsum(stat.btc_earned), name='BTC, if held', short_name='btc_held')

        stat.istat.add_account(fs.arr.multiply(stat.btc_held, env.btc_price), name='BTC Value, if held', short_name='btc_value_held')

        # roi = ROITemplate(stat, mine)
        # stat.add_related(roi.short_name, roi)

# class ROITemplate:
#     def __new__(self, stat, mine):
#         resamp = stat.istat.resample('M').sum(last=['btc_held', 'btc_value_held'])
#         roi_periods = pd.period_range(end=resamp.periods[-1], periods=resamp.periods.size + 1, freq=resamp.periods.freq)

#         roi = fs.FinancialStatement(name='ROI', periods=roi_periods)
#         outlays = np.zeros(roi_periods.size)
#         outidx = mine.implement.start
#         outlays[outidx] = -mine.capital_cost

#         inflows = np.zeros(roi_periods.size)
#         inflows[1:] = resamp.op_flow_sold.values

#         btc = np.zeros(roi_periods.size)
#         btc[1:] = resamp.btc_held.values

#         cum_btc = np.zeros(roi_periods.size)
#         held_delta = np.zeros(roi_periods.size)
#         held_delta[1:] = (resamp.btc_value_held - resamp.btc_value_held.shift(1)).values
#         held_delta[1] = resamp.btc_value_held.iloc[0]

#         cum_btc[1:] = resamp.btc_value_held.values

#         roi.add_account(outlays, name='Cash Outlays', short_name='cash_out')
#         roi.add_account(held_delta, name='Operating Cash Flow, held', short_name='op_flow_held')
#         roi.add_account(inflows, name='Operating Cash Flow, sold', short_name='op_flow_sold')
#         roi.add_account(btc, name='BTC Held', short_name='btc_held', hide=True)
#         roi.add_account(cum_btc, name='Cumulative BTC Value, held', short_name='cum_btc_value_held')

#         with roi.add_metrics() as am:
#             am(roi.cash_out + roi.op_flow_sold, name='Net Cash Flow, sold', short_name='net_flow_sold')
#             am(roi.cash_out + roi.op_flow_held, name='Net Cash Flow, held', short_name='net_flow_held')
#             am(fs.cumsum(roi.net_flow_sold), name='Cumulative Net, sold', short_name='cum_flow_sold')
#             am(fs.cumsum(roi.net_flow_held), name='Cumulative Net, held', short_name='cum_flow_held')

#             am(roi.cum_flow_sold / -outlays[outidx], name='ROI, sold', short_name='roi_sold')
#             am(roi.cum_flow_held / -outlays[outidx], name='ROI, held', short_name='roi_held')

#         return roi

# class ProjectStats(PoolStats):
#     def __new__(cls, mines, periods, pbar=None, **kwargs):
#         minerstats = []
        
#         for mine in mines:
#             minerstat = ProjectTemplate(mine, periods=periods, no_model=True, **kwargs)
#             minerstats.append(minerstat)
#             if pbar is not None:
#                 pbar.update(1)

#         assert np.all(minerstats[0].lineitems.short_names == [s.lineitems.short_names for s in minerstats])
        
#         obj = np.asarray([o for o in minerstats], dtype='object').view(cls)
#         obj.mines = mines
#         obj.periods = periods

#         return obj

#     @property
#     def rois(self):
#         has_roi = [hasattr(mstat, 'roi') for mstat in self]
#         rois = [mstat.roi for mstat in self[has_roi]]
#         return ROIS(rois, self.mines[has_roi])

#     def finalize(self, env, pbar=None):
#         for mine, projstat in self.items():
#             ProjectTemplate.finalize(projstat, mine, env)
#             if pbar:
#                 pbar.update()
                
# class ROIS(ProjectStats):
#     def __new__(cls, rois, mines):
#         obj = np.asarray([o for o in rois], dtype='object').view(cls)
#         obj.mines = mines

#         return obj

#     def __array_finalize__(self, obj):
#         if obj is None: return
#         self.mines = getattr(obj, 'mines', None)

#     @property
#     def btc_held(self):
#         return [roi.btc_held[-1] for roi in self]

#     @property
#     def op_flow_sold(self):
#         return [roi.op_flow_sold.sum() for roi in self]

#     @property
#     def op_flow_held(self):
#         return [roi.op_flow_held.sum() for roi in self]

#     @property
#     def total_cash_sold(self):
#         return [roi.cum_flow_sold[-1] for roi in self]

#     @property
#     def total_cash_held(self):
#         return [roi.cum_flow_held[-1] for roi in self]

#     @property
#     def rois_sold(self):
#         return [roi.roi_sold[-1] for roi in self]

#     @property
#     def rois_held(self):
#         return [roi.roi_held[-1] for roi in self]

#     def irr(self, mine, roi, n=None, lineitem='net_flow_held'):
#         lineitem = getattr(roi, lineitem)
#         if n is None:
#             n = roi.periods.size
        
#         start = mine.implement.start
#         return npf.irr(lineitem.iloc[start: start + n])

#     def irrs(self, periods, **kwargs):
#         return [(1 + self.irr(mine, roi, periods, **kwargs))**12 - 1 for mine, roi in self.items()]

#     @property
#     def three_yr_irr(self):
#         return self.irrs(36)

#     @property
#     def five_yr_irr(self):
#         return self.irrs(60)

#     @property
#     def terminal_irr(self):
#         return self.irrs(None)

#     @property
#     def breakevens(self):
#         def beven_idx(mine, ser):
#             start = mine.implement.start
#             return ser.iloc[start:].abs().idxmin()

#         return [beven_idx(mine, roi.roi_held) for mine, roi in self.items()]

#     def summary(self):
#         return pd.DataFrame([
#             self.btc_held,
#             self.op_flow_held,
#             # self.total_cash_sold,
#             self.total_cash_held,
#             # self.rois_sold,
#             self.rois_held,
#             self.three_yr_irr,
#             self.five_yr_irr,
#             self.terminal_irr,
#             self.breakevens,
#             ], index=[
#                 'BTC, held', 
#                 'Net Cash Flow, held',
#                 'Net Gain, held', 
#                 'ROI, held',
#                 'IRR 3-year, held', 'IRR 5-year held', f'IRR terminal, held',
#                 'Breakeven'
#             ],
#             columns=self.mines.names
#         )