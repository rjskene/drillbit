import string
import numpy as np
import pandas as pd

from bitcoin.helpers import pbar_update
from bitcoin.style import grc_style
from bitcoin.charts import charts
from bitcoin.excel import excel_columns
from envirosim.excel import EnviroModelMaker, on_import
from .statements import ProjectStats

import btc
def efficiency_summary(State):
    arr = np.zeros((10, State.projstats.size))
    for i, (mine, stat) in enumerate(zip(State.mines.projects, State.projstats)):
        tot_hashes = btc.units.Hashes(stat.env.hashes.sum())
        tot_energy = btc.units.Energy(stat.env.energy.sum())
        eff = btc.units.Efficiency(tot_energy.in_joules() / tot_hashes)

        energy_exp = stat.istat.energy_exp.sum()
        tot_exp = (stat.istat.energy_exp + stat.istat.ops + stat.istat.prop_tax + stat.istat.taxes).sum()
        
        energy_hp = btc.units.HashPrice(energy_exp / tot_hashes)
        opex_hp = btc.units.HashPrice(tot_exp / tot_hashes)
        capex_hp = btc.units.HashPrice(mine.capital_cost / tot_hashes)
        cost_hp = btc.units.HashPrice((mine.capital_cost + tot_exp) / tot_hashes)

        block_rev = State.env.mkt_rev.mean() * stat.env.win_per.mean()
        block_cost = btc.units.HashRate(stat.env.hr.mean()).hashes_per_block() * btc.units.HashPrice(cost_hp)
        block_net = block_rev - block_cost
        
        arr[:, i] = [tot_hashes, tot_energy, eff, energy_hp, opex_hp, capex_hp, cost_hp, block_rev, block_cost, block_net]        

    spacer = np.zeros(State.projstats.size)*np.nan
    idx = [
        'Totals',
        'Total Hashes',
        'Energy Consumption',
        'Efficiency',
        'Hash Costs ($ / EH)',
        'Energy Expense',
        'Cash Expenses',
        'Capital Cost',
        'Total Cost',
        'Average Block Economics',
        'Revenue per Block',
        'Cost per Block',
        'Net per Block'
    ]
    return pd.DataFrame(np.vstack((spacer, arr[:3], spacer, arr[3:-3], spacer, arr[-3:])), columns=State.mines.projects.names, index=idx)

def update_proj_summ_table(State):
    ws = State.wb.sheets[State.WS['proj_comp']]
    ws.range('A2:AA40').clear_contents()
    ws.range('A1').unmerge()

    summ = State.mines.projects.summary() 
    summ.loc['Power':'Total Power'] /= 1e6
    summ.loc['Peak Hash Rate'] /= 1e18
    ws.range('A2').options(index=True, headers=True).value = summ

    colidx = summ.shape[1]
    col = string.ascii_uppercase[colidx]
    colnxt = string.ascii_uppercase[colidx + 1]
    ws.range(f'A1:{col}1').merge()

    summ = State.projstats.rois.summary()
    summ.loc['Breakeven'] = pd.PeriodIndex(summ.loc['Breakeven']).strftime('%Y-%b')
    ws.range('A18').options(pd.DataFrame, index=True, headers=True).value = summ

    eff_summ = efficiency_summary(State)
    eff_summ.loc['Total Hashes'] /= 1e24
    eff_summ.loc['Energy Consumption'] /= 1e12
    eff_summ.loc['Efficiency'] *= 1e12
    eff_summ.loc['Hash Costs ($ / EH)':'Total Cost'] *= 1e18
    ws.range('A27').options(pd.DataFrame, index=True, headers=True).value = eff_summ

    clean_rng = f'{colnxt}1:AA40'
    ws.range(clean_rng).clear_formats()
    ws.range(clean_rng).color = (255,255,255)

    ws.range('B2:B40').copy()
    ws.range(f'C2:{col}40').paste("formats")

def fill_project_sheet(state, mstat, roi, after, cols):
    if mstat.name not in [s.name for s in state.wb.sheets]:
        ws = state.wb.sheets.add(mstat.name, after=after)
        ws.api.Tab.Color = grc_style.vlblue.bgrhex
        ws_temp = state.wb.sheets[state.WS['proj_temp']]
        ws_temp.cells.copy()
        ws.cells.paste("formats")
    else:
        ws = state.wb.sheets[mstat.name]

    after = mstat.name
        
    resamp = mstat.env.resample('Q').sum(last=['n_miners'], mean=['hr', 'win_per'])
    ws.range('A1').options(index=True, header=True).value = resamp.to_frame(with_periods=False)
    ws.range('A14').options(index=True, header=True).value = mstat.istat.resample('Q').sum(last=['btc_held', 'btc_value_held']).to_frame(with_periods=False)
    ws.range('A38').options(index=True, header=True).value = roi.resample('Q').last(summ=['cash_out', 'op_flow_sold', 'net_flow_sold']).to_frame(with_periods=False)
    
    lastcolidx = resamp.shape[1]
    col = cols[lastcolidx]
    colnxt = cols[lastcolidx + 1]

    clean_rng = f'{colnxt}1:XFD100'
    ws.range(clean_rng).clear_formats()
    ws.range(clean_rng).clear_contents()
    ws.range(clean_rng).color = (255,255,255)

    ws.range('B1:B46').copy()
    ws.range(f'C1:{col}46').paste("formats")
    
    return after

def del_profile_sheets(state):
    sheet_names = np.array([s.name for s in state.wb.sheets])
    start = np.argwhere(sheet_names == state.WS['proj_comp'])[0,0] + 1
    last = np.argwhere(sheet_names == 'Support --->')[0,0]
    sheets_to_del = np.setdiff1d(sheet_names[start:last], state.projstats.names)

    for name in sheets_to_del:
        state.wb.sheets(name).delete()

class GenericModelMaker(EnviroModelMaker):
    def generate_statements(self, mines, pbar=None, **kwargs):
        super().generate_statements(mines, pbar=pbar, **kwargs)

        projstats = ProjectStats(self.state.mines.projects, self.state.periods, pbar=pbar)
        projstats.finalize(self.state.env, pbar=pbar) # .env added to state in super() call above
        self.state.set_project_statements(projstats)

    def insert_statements(self, pbar, **kwargs):
        super().insert_statements(**kwargs)

        if not self.state.environment_only:
            with pbar_update(pbar, desc='Update summary comparison...') as p:
                update_proj_summ_table(self.state)

            with pbar_update(pbar, desc='Inserting profile financial statements...', update=False) as p:
                cols = excel_columns()
                after = 'Project Comparison'
                for mstat, roi in zip(self.state.projstats.projects, self.state.projstats.rois):
                    with pbar_update(pbar, desc='Inserting profile financial statements...') as p:
                        after = fill_project_sheet(self.state, mstat, roi, after, cols)

                with pbar_update(pbar, desc='Removing old projects...') as p:
                    del_profile_sheets(self.state)

    def create_charts(self, **kwargs):
        pbar = kwargs.pop('pbar', None)
        chart_sheet = kwargs.pop('chart_sheet')
        
        if not self.state.environment_only:
            charts.chart_miner_comps(self.state, self.state.get_ws('proj_comp'))
            if pbar is not None:
                pbar.update(1)

            charts.chart_hr_project_comps(self.state, self.state.get_ws('proj_comp'))
            if pbar is not None:
                pbar.update(1)

            charts.chart_energy_project_comps(self.state, self.state.get_ws('proj_comp'))
            if pbar is not None:
                pbar.update(1)

            charts.chart_mined_energy_project_comps(self.state, self.state.get_ws('proj_comp'))
            if pbar is not None:
                pbar.update(1)

            charts.chart_win_per_project_comps(self.state, self.state.get_ws('proj_comp'))
            if pbar is not None:
                pbar.update(1)

            charts.chart_btc_mined_project_comps(self.state, self.state.get_ws('proj_comp'))
            if pbar is not None:
                pbar.update(1)

            charts.chart_gm_project_comps(self.state, self.state.get_ws('proj_comp'))
            if pbar is not None:
                pbar.update(1)

            charts.chart_btc_value_held_project_comps(self.state, self.state.get_ws('proj_comp'))
            if pbar is not None:
                pbar.update(1)

            charts.chart_roi_project_comps(self.state, self.state.get_ws('proj_comp'))
            if pbar is not None:
                pbar.update(1)

        super().create_charts(pbar=pbar, chart_sheet=chart_sheet, **kwargs)
