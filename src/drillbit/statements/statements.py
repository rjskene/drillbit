from copy import deepcopy
import numpy as np
import pandas as pd
import numpy_financial as npf

from .funcs import total_energy, expected_difficulty, hashes_to_hash_rate, \
    hash_rate_to_hashes, win_percentage, staggered_amortize, amortize
from ..__new_units__ import AbstractBaseUnit, Energy

import finstat as fs

def init_environment(block_schedule, price, fees, hash_rate):
    stat = fs.FinancialStatement(name='BTC Environment', periods=block_schedule.index)

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
    def __new__(self, env, project):
        """
        Attributes from the project, rig, and infrastructure objects are plugged into the
        financial statement template. These attributes may either be:
            + a single value or an, or
            + an array of values. 
            
        If an array of values is provided, the array must be the same length as the periods array.

        If a single value is provided, the value will be broadcast to the length of the periods array.

        Parameters
        ----------
        project : Project
        """
        stat = fs.FinancialStatement(name=project.name, periods=env.periods)
        stat.add_factor('tax_rate', project.tax_rate)
        stat.add_statement(name='Environment', short_name='env')
        stat.add_statement(name='Income Statement')

        stat.env.add_account(project.rigs.schedule, name='Number of Miners', short_name='n_miners')
        stat.env.add_account(
            fs.arr.multiply(stat.n_miners, project.consumption_per_rig_per_block.in_joules()), 
            name='Energy (J) - Miner', 
            short_name='miner_energy_in_joules',
            hide=True
        )
        stat.env.add_account(
            fs.arr.multiply(stat.n_miners, project.consumption_per_rig_per_block), 
            name='Energy - Miner', 
            short_name='miner_energy'
        )
        stat.env.add_account(total_energy(stat.miner_energy, project.pue), name='Energy - Infra', short_name='infra_energy')
        stat.env.add_account(fs.arr.add(stat.infra_energy, stat.miner_energy), name='Energy')

        stat.env.add_account(fs.arr.multiply(project.hash_rate_per_rig.value, stat.n_miners), name='Hash Rate')
        stat.env.add_account(hash_rate_to_hashes(stat.hr), name='Hashes')
        stat.env.add_account(win_percentage(stat.hashes, env.difficulty), name='Hash Share', short_name='hash_share')

        stat.env.add_account(fs.arr.multiply(stat.hash_share, env.reward), name='BTC Reward', short_name='btc_reward')
        stat.env.add_account(fs.arr.multiply(stat.hash_share, env.fees), name='Transaction Fees', short_name='traxn_fees')
        stat.env.add_account(fs.arr.multiply(stat.btc_reward, project.pool_fees), name='Pool Fees (\u0243)', short_name='pool_fees_in_btc')
        stat.env.add_account(fs.arr.add(stat.btc_reward, stat.traxn_fees, -stat.pool_fees_in_btc), name='BTC Mined', short_name='btc_mined')

        stat.istat.add_account(fs.arr.multiply(stat.btc_reward, env.btc_price), name='Revenue - Reward', short_name='reward_rev')
        stat.istat.add_account(fs.arr.multiply(stat.traxn_fees, env.btc_price), name='Revenue - Fees', short_name='fee_rev')
        stat.istat.add_account(fs.arr.add(stat.fee_rev, stat.reward_rev), name='Gross Revenue', short_name='gross_rev')

        stat.istat.add_account(fs.arr.multiply(stat.pool_fees_in_btc, env.btc_price), name='Pool Fees', short_name='pool_fees')
        stat.istat.add_account(fs.arr.add(stat.gross_rev, -stat.pool_fees), name='Net Revenue', short_name='net_rev')
        stat.istat.add_account(fs.arr.multiply(stat.btc_mined, env.btc_price), name='Test Net Revenue', short_name='test_net_rev', hide=True)

        stat.istat.add_account(fs.arr.multiply(stat.energy, project.energy_price), name='Energy Expenses', short_name='energy_exp')
        stat.istat.add_account(fs.arr.add(stat.reward_rev, -stat.energy_exp), name='Gross Profit')
        stat.istat.add_account(fs.arr.divide(stat.gp, stat.net_rev), name='Gross Margin', hide=True)

        stat.istat.add_account(project.opex, name='Operations', short_name='ops')
        stat.istat.add_account(project.property_tax, name='Property Tax', short_name='prop_tax')

        stat.istat.add_account(fs.arr.add(stat.gp, -stat.ops, -stat.prop_tax), name='EBITDA')

        stat.istat.add_account(
            staggered_amortize(
                env.periods.size,
                project.rigs.amortization,
                project.rigs.price,
                stat.n_miners,
            ), name='Rig Amortization', short_name='rig_amort'
        )
        for i, infra in enumerate(project.infrastructure):
            stat.istat.add_account(
                amortize(
                    env.periods.size,
                    infra.amortization,
                    infra.price,
                    infra.quantity,
                    periods=env.periods,
                ), name=f'Infra {infra.name} Amortization', short_name=f'infra_{i}_amort'
            )
        stat.add_account(
            fs.arr.add(*[getattr(stat, f'infra_{i}_amort') for i in range(len(project.infrastructure))]),
            name='Infra Amortization', short_name='infra_amort'
        )
        # stat.istat.add_account(
        #     amortize(
        #         env.periods.size,

        #     ), name='Building Amortization', short_name='build_amort'
        # )
        # , stat.cool_amort
        stat.istat.add_account(fs.arr.add(stat.rig_amort, stat.infra_amort), name='Depreciation for Taxes', short_name='tax_depn')

        stat.istat.add_account(fs.arr.add(stat.ebitda, -stat.tax_depn), name='EBIT')
        stat.istat.add_account(fs.arr.multiply(stat.ebit, stat.tax_rate), name='Taxes')

        stat.istat.add_account(fs.arr.add(stat.ebit, -stat.taxes), name='Profit, if sold', short_name='profit_sold')
        stat.istat.add_account(fs.arr.add(stat.ebitda, -stat.taxes), name='Operating Cash Flow, if sold', short_name='op_flow_sold')

        stat.istat.add_account(fs.arr.add(stat.energy_exp, stat.ops, stat.prop_tax, stat.taxes), name='Cash Expenses', short_name='cash_exp', hide=True)
        stat.istat.add_account(fs.arr.divide(stat.cash_exp, env.btc_price), name='BTC Converted for Expenses', short_name='converted')

        stat.istat.add_account(fs.arr.add(stat.btc_mined, -stat.converted), name='BTC Earned', short_name='btc_earned')
        stat.istat.add_account(fs.arr.cumsum(stat.btc_earned), name='BTC, if held', short_name='btc_held')

        stat.istat.add_account(fs.arr.multiply(stat.btc_held, env.btc_price), name='BTC Value, if held', short_name='btc_value_held')

        roi = ROITemplate(stat, project)
        stat.add_related(roi.short_name, roi)

        return stat

class ROITemplate:
    def __new__(self, stat, project):
        resamp = stat.istat.resample('M').sum(last=['btc_held', 'btc_value_held'])
        roi_periods = pd.period_range(
            end=resamp.periods[-1],
            periods=resamp.periods.size + 1, 
            freq=resamp.periods.freq
        )

        roi = fs.FinancialStatement(name='ROI', periods=roi_periods)
        outlays = np.zeros(roi_periods.size)
        outidx = 0
        outlays[outidx] = -project.capital_cost()

        inflows = np.zeros(roi_periods.size)
        inflows[1:] = resamp.op_flow_sold.values

        btc = np.zeros(roi_periods.size)
        btc[1:] = resamp.btc_held.values

        cum_btc = np.zeros(roi_periods.size)
        held_delta = np.zeros(roi_periods.size)
        held_delta[1:] = (resamp.btc_value_held - resamp.btc_value_held.shift(1)).values
        held_delta[1] = resamp.btc_value_held.iloc[0]

        cum_btc[1:] = resamp.btc_value_held.values

        rig_out = np.zeros(roi_periods.size)
        outidx = 0
        rig_out[outidx] = -project.rig_cost()
        roi.add_account(rig_out, name='Rigs Outlay', short_name='rigs_out')

        for k, v in project.infra_cost_schedule().items():        
            infra_out = np.zeros(roi_periods.size)
            outidx = 0
            infra_out[outidx] = -v
            short_name = k.replace(" ", "_").replace('/', '_').replace('.', '_').replace('-', '_').lower()
            roi.add_account(infra_out, name=f'{k} Outlay', short_name=f'{short_name}_out')

        infra_out = np.zeros(roi_periods.size)
        outidx = 0
        infra_out[outidx] = -project.infra_cost()
        roi.add_account(infra_out, name='Infrastructure Outlay', short_name='infra_out')

        build_out = np.zeros(roi_periods.size)
        outidx = 0
        build_out[outidx] = -project.building_cost()
        roi.add_account(build_out, name='Building Outlay', short_name='build_out')

        roi.add_account(outlays, name='Cash Outlays', short_name='cash_out')
        roi.add_account(held_delta, name='Operating Cash Flow, held', short_name='op_flow_held')
        roi.add_account(inflows, name='Operating Cash Flow, sold', short_name='op_flow_sold')
        roi.add_account(btc, name='BTC Held', short_name='btc_held', hide=True)
        roi.add_account(cum_btc, name='Cumulative BTC Value, held', short_name='cum_btc_value_held')

        with roi.add_metrics() as am:
            am(roi.cash_out + roi.op_flow_sold, name='Net Cash Flow, sold', short_name='net_flow_sold')
            am(roi.cash_out + roi.op_flow_held, name='Net Cash Flow, held', short_name='net_flow_held')
            am(fs.cumsum(roi.net_flow_sold), name='Cumulative Net, sold', short_name='cum_flow_sold')
            am(fs.cumsum(roi.net_flow_held), name='Cumulative Net, held', short_name='cum_flow_held')

            am(roi.cum_flow_sold / -outlays[outidx], name='ROI, sold', short_name='roi_sold')
            am(roi.cum_flow_held / -outlays[outidx], name='ROI, held', short_name='roi_held')

        return roi

class analysis:
    def __init__(self, stat, project):
        self.stat = stat
        self.project = project

    @property
    def roi(self):
        return self.stat.roi

    @property
    def btc_held(self):
        return self.roi.btc_held[-1]

    @property
    def op_flow_sold(self):
        return self.roi.op_flow_sold.sum()

    @property
    def op_flow_held(self):
        return self.roi.op_flow_held.sum()

    @property
    def total_cash_sold(self):
        return self.roi.cum_flow_sold[-1]

    @property
    def total_cash_held(self):
        return self.roi.cum_flow_held[-1]

    @property
    def roi_sold(self):
        return self.roi.roi_sold[-1]

    @property
    def roi_held(self):
        return self.roi.roi_held[-1]

    def irr(self, n=None, lineitem='net_flow_held', annualize=True):
        lineitem = getattr(self.roi, lineitem)
        if n is None:
            n = self.roi.periods.size
        
        start = 0
        irr = npf.irr(lineitem.iloc[start: start + n])

        if annualize:
            irr = (1 + irr)**12 - 1
        
        return irr

    @property
    def three_yr_irr(self):
        return self.irr(36)

    @property
    def five_yr_irr(self):
        return self.irr(60)

    @property
    def terminal_irr(self):
        return self.irr(None)

    @property
    def breakeven(self):
        return self.roi.roi_held.abs().idxmin()

    @property
    def hashes(self):
        return self.stat.env.hashes.sum()

    @property
    def energy(self):
        return Energy(self.stat.env.energy.sum()).in_joules()
    
    @property
    def efficiency(self):
        return self.energy / self.hashes # energy must be in joules for this to work / returns joules/hash

    @property
    def energy_expense(self):
        return self.stat.istat.energy_exp.sum()

    @property
    def total_cash_expense(self):
        return self.stat.istat.energy_exp.sum() + self.stat.istat.ops.sum() + self.stat.istat.prop_tax.sum() + self.stat.istat.taxes.sum()

    @property
    def rig_costs(self):
        return -self.stat.roi.rigs_out.sum()

    @property
    def infra_costs(self):
        return {infra.name + ' Cost': infra.cost() for infra in self.project.infrastructure}

    @property
    def total_infra_costs(self):
        return sum(self.infra_costs.values())

    @property
    def capital_costs(self):
        return -self.stat.roi.cash_out.sum()

    @property
    def total_cost(self):
        return self.total_cash_expense + self.capital_costs

    @property
    def hash_prices(self):
        hash_prices = {}
        for n in [n for n in self.stat.roi.G.nodes if '_out' in n]:
            acct = getattr(self.stat.roi, n)
            hash_prices[n.rstrip('_out').replace('_', ' ').title() + ' Hash Price'] = -acct.sum() / self.hashes

        return hash_prices

    def summary(self):
        return {
            'Capacity': self.project.capacity,
            'Compute Power': self.project.compute_power,
            'Infra Power': self.project.infra_power,
            'Number of Rigs': self.project.rigs.quantity,
            'Hash Rate': self.project.rigs.total_hash_rate,
            'Hash Rate per Rig': self.project.hash_rate_per_rig,
            'Total Hashes': self.hashes,
            'Energy Consumption': self.energy,
            'Efficiency': self.efficiency,
            'Energy Expense': self.energy_expense,
            'Total Expenses': self.total_cash_expense,
            'Rig Costs': self.rig_costs,
            **self.infra_costs,
            'Total Infra Costs': self.total_infra_costs,
            'Capital Costs': self.capital_costs,
            'Total Cost': self.total_cost,
            **self.hash_prices,
            'BTC, held': self.btc_held,
            'Net Cash Flow, held': self.op_flow_held,
            'Net Gain, held': self.total_cash_held,
            'ROI, held': self.roi_held,
            'IRR 3-year, held': self.three_yr_irr,
            'IRR 5-year, held': self.five_yr_irr,
            f'IRR terminal, held': self.terminal_irr,
            'Breakeven': self.breakeven.strftime('%Y-%m-%d'),
        }