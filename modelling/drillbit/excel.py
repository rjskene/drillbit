import string
import itertools as it
import numpy as np
import pandas as pd
import win32api
import win32con

from tqdm.auto import tqdm
from xlwings.utils import rgb_to_int
from xlwings.conversion import Converter, PandasDataFrameConverter

import btc
from btc import sim

from .charts import charts
from .style import grc_style
from .helpers import pbar_update
from .statements import PoolStats, ProjectStats, init_enviro

def excel_columns():
    abeta = list(string.ascii_uppercase)
    abeta2 = [''.join(pair) for pair in it.product(abeta, abeta)]
    abeta3 = [''.join(trip) for trip in it.product(abeta, abeta, abeta)]

    return abeta + abeta2 + abeta3

def msgbox(wb, msg, kind='Info'):
    win32api.MessageBox(
        wb.app.hwnd, msg, kind,
        win32con.MB_ICONINFORMATION
    )

def btn_warning(ws, btn_name, msg):
    btn = ws.api.OLEObjects(btn_name)
    btn.Object.Caption = msg
    btn.Object.BackColor = rgb_to_int((255,0,0))

def button_update(state, ws, model, update_checklist=True):
    ws.api.OLEObjects(state.BUTTONS[model]).Object.BackColor = rgb_to_int(grc_style.vvlblue.hex_to_rgb())
    ws.api.OLEObjects(state.BUTTONS[model]).Object.Caption = 'Updated!'

    if update_checklist:
        state.wb.sheets[state.WS['meta']].range(state.CHECKLIST_CELLS[model]).value = 1

def pass_checklist(state):
    for k, cell in state.CHECKLIST_CELLS.items():
        state.wb.sheets[state.WS['meta']].range(cell).value = 1

class DataFrameDropna(Converter):

    base = PandasDataFrameConverter

    @staticmethod
    def read_value(builtin_df, options):
        return builtin_df.dropna()

def check_ready(state):
    ws = state.wb.sheets[state.WS['meta']]
    cells = list(state.CHECKLIST_CELLS.values())
    rng = f'{cells[0]}:{cells[-1]}'
    checks = sum(ws.range(rng).value)

    passed_check = checks == 3
    if not passed_check:
        msgbox(state.wb, 'You must complete the checklist.')

    return passed_check

def profiles_to_load(profile_type, State):
    sheet_name = State.WS[profile_type]
    loadcell = State.LOAD_CELLS[profile_type]

    datadir = State.parent_path.parent / 'data' / profile_type
    profiles = [' '.join(file_.name.split('.')[0].title().split('_')) for file_ in datadir.iterdir() if '~' not in file_.name]
    State.wb.sheets[sheet_name].range(loadcell).api.Validation.Delete()
    State.wb.sheets[sheet_name].range(loadcell).api.Validation.Add(Type=3,Formula1=','.join(profiles))

def load_profiles(profile_type, State):
    sheet_name=State.WS[profile_type] 
    table_name=State.TABLES[profile_type] 
    loadcell=State.LOAD_CELLS[profile_type]
    
    ws = State.wb.sheets[sheet_name]
    filename = ws.range(loadcell).value
    filename = '_'.join(filename.lower().split(' '))
    mine_profiles = State.parent_path.parent / 'data' / profile_type / f'{filename}.xlsx'
    df = pd.read_excel(mine_profiles)

    State.wb.app.enable_events = False
    ws.range(f'{table_name}').clear_contents()
    State.wb.app.enable_events = True
    ws.range(f'{table_name}').options(pd.DataFrame, header=False, index=False).value = df
    print (f'{profile_type.title()} update.')

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
    summ.loc['Power':'Power for Cooling'] /= 1e6
    summ.loc['Peak Hash Rate':'Hash Rate per Miner'] /= 1e18
    summ.loc['Hash Rate per Miner'] *= 1e6
    ws.range('A2').options(index=True, headers=True).value = summ

    colidx = summ.shape[1]
    col = string.ascii_uppercase[colidx]
    colnxt = string.ascii_uppercase[colidx + 1]
    ws.range(f'A1:{col}1').merge()

    summ = State.projstats.rois.summary()
    summ.loc['Breakeven'] = pd.PeriodIndex(summ.loc['Breakeven']).strftime('%Y-%b')
    ws.range('A20').options(pd.DataFrame, index=True, headers=True).value = summ

    eff_summ = efficiency_summary(State)
    eff_summ.loc['Total Hashes'] /= 1e24
    eff_summ.loc['Energy Consumption'] /= 1e12
    eff_summ.loc['Efficiency'] *= 1e12
    eff_summ.loc['Hash Costs ($ / EH)':'Total Cost'] *= 1e18
    ws.range('A29').options(pd.DataFrame, index=True, headers=True).value = eff_summ

    clean_rng = f'{colnxt}1:AA40'
    ws.range(clean_rng).clear_formats()
    ws.range(clean_rng).color = (255,255,255)

    ws.range('B2:B42').copy()
    ws.range(f'C2:{col}42').paste("formats")

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

class GenericModelMaker:
    def __init__(self, state):
        self.state = state

    def generate_block_schedule(self):
        ws = self.state.wb.sheets[self.state.WS['block_sched']]   
        sched = self.state.generate_block_schedule()

        ws.range(self.state.TABLES['block_sched']).options(index=True, header=True).value = sched.iloc[:100]
        button_update(self.state, ws, 'block_sched')
        charts.chart_block_sched(self.state, ws)

    def generate_btc_forecast(self):
        ws = self.state.get_ws('btc_price')
        btc_price = self.state.generate_btc_forecast()
        frame = btc_price.to_frame()
        frame.columns = ['Price']

        ws.range(self.state.TABLES['btc_price']).options(index=True, header=True).value = frame.iloc[:100]
        button_update(self.state, ws, 'btc_price')
        
        charts.chart_btc_forecast(self.state, ws)
        charts.chart_btc_price(self.state, ws)

    def generate_fee_forecast(self):
        ws = self.state.wb.sheets[self.state.WS['fees']]

        traxn_fees = self.state.generate_fee_forecast()
        frame = traxn_fees.to_frame()
        frame.columns = ['Fees']

        ws.range(self.state.TABLES['fees']).options(index=True, header=True).value = frame.iloc[:100]
        button_update(self.state, ws, 'fees')
        
        charts.chart_fee_forecast(self.state, ws)
        charts.chart_fee_price(self.state, ws)

    def create_environment(self, complete_msg=True, pbar_kws={}, **kwargs):
        update_charts = kwargs.pop('update_charts', self.state.UPDATE_CHARTS)
        wipe_stats = kwargs.pop('wipe_stats', False)
        if wipe_stats:
            self.state.wipe()

        kwargs['btn_sheet'] = self.state.get_ws(kwargs.get('btn_sheet', 'Simulator'))
        kwargs['tracker_sheet'] = self.state.get_ws(kwargs.get('tracker_sheet', 'BTC Meta'))
        kwargs['chart_sheet'] = self.state.get_ws(kwargs.get('chart_sheet', 'Mining Environment'))

        validated = self.validation(**kwargs)
        if validated:
            with tqdm(**pbar_kws) as pbar:
                pbar.set_description('Initializing...')
                env = init_enviro(self.state.block_sched, self.state.traxn_fees, self.state.btc_price)

                with pbar_update(pbar, desc='Creating environment data...') as p:
                    if self.state.ENV_ONLY: # whether to include only Pools or both Pools & Projects in environment
                        mines = self.state.mines.pools
                    else:
                        mines = self.state.mines

                    financials = self.simulate_mining_environment(env, mines)

                with pbar_update(pbar, desc='Generating mining statements...') as p:
                    self.generate_pool_statements(mines, financials=financials, env=env, pbar=pbar)
                    self.insert_env_statement(pbar=pbar, **kwargs)

                if update_charts:
                    pbar.set_description('Creating charts...')
                    self.create_env_charts(pbar=pbar, **kwargs)

                with pbar_update(pbar, desc='Update formats...') as p: 
                    self.update_formats(pbar=pbar, **kwargs)
   
                pbar.set_description('Environment Sim Complete.')

        if complete_msg:
            msgbox(self.state.wb, 'Environment Simulation Complete.')

    def validation(self, *args, **kwargs):
        return check_ready(self.state)

    def simulate_mining_environment(self, env, mines):
        ends = sim.retgt_ranges(self.state.block_sched, self.state.BTC.RETARGET_BLOCKS)
        financials, _ = sim.simulate(mines, self.state.block_sched, env.mkt_rev.values, ends, pbar_kws={'leave': False})
        return financials

    def generate_pool_statements(self, mines, pbar=None, **kwargs):
        financials = kwargs.pop('financials')
        env = kwargs.pop('env')

        self.state.mines.implement(self.state.periods.size)
        minerstats = PoolStats(mines, financials[:, sim.ARRAY_KEY['miner_energy']], financials[:, sim.ARRAY_KEY['hashes']], self.state.periods, pbar=pbar)
        minerstats.finalize(env, financials[:, sim.ARRAY_KEY['hashes']].sum(axis=0), pbar=pbar)
        self.state.set_global_environment(env)
        self.state.set_mine_statements(minerstats)

    def insert_env_statement(self, pbar=None, **kwargs):
        chart_sheet = kwargs.pop('chart_sheet')
        chart_sheet.range('A1').options(header=True, index=True).value = self.state.env.to_frame(with_periods=False).T.iloc[:100]

    def create_env_charts(self, **kwargs):
        pbar = kwargs.pop('pbar', None)
        chart_sheet = kwargs.pop('chart_sheet')
        op, count, gm, halve_ticks, hr = charts.env_sim_pre_processing(self.state)

        charts.chart_operational_mines(op, halve_ticks, chart_sheet, self.state.parent_path)
        if pbar is not None:
            pbar.update(1)

        charts.mine_gross_margin(gm, halve_ticks, chart_sheet, self.state.parent_path)
        if pbar is not None:
            pbar.update(1)

        charts.mine_hash_rate(hr, halve_ticks, chart_sheet, self.state.parent_path)
        if pbar is not None:
            pbar.update(1)

        charts.count_operational(count, chart_sheet, self.state.parent_path)
        if pbar is not None:
            pbar.update(1)

        charts.chart_mining_revenue(self.state, chart_sheet)
        if pbar is not None:
            pbar.update(1)

        charts.chart_mining_revenue_comp(self.state, chart_sheet)
        if pbar is not None:
            pbar.update(1)

        charts.chart_difficulty(self.state, chart_sheet)
        if pbar is not None:
            pbar.update(1)

        charts.chart_difficulty_comp(self.state, chart_sheet)
        if pbar is not None:
            pbar.update(1)

        charts.chart_hash_rate(self.state, chart_sheet)
        if pbar is not None:
            pbar.update(1)

        charts.chart_hash_rate_comp(self.state, chart_sheet)

    def update_formats(self, **kwargs):
        btn_sheet = kwargs.pop('btn_sheet')
        btn_name = kwargs.pop('btn_name')

        btn_sheet.api.OLEObjects(btn_name).Object.BackColor = rgb_to_int(grc_style.vvlblue.hex_to_rgb())
        btn_sheet.api.OLEObjects(btn_name).Object.Caption = 'Updated!'

    def create_projects(self, complete_msg=True, pbar_kws={}, **kwargs):
        update_charts = kwargs.pop('update_charts', self.state.UPDATE_CHARTS)
        insert_stats = kwargs.pop('insert_stats', self.state.INS_STATS)
        kwargs['btn_sheet'] = self.state.get_ws(kwargs.get('btn_sheet'))

        with tqdm(**pbar_kws) as pbar:
            with pbar_update(pbar, desc='Generating mining statements...') as p:
                self.generate_project_statements(pbar=pbar, **kwargs)

            if update_charts:
                pbar.set_description('Creating charts...')
                self.create_project_charts(pbar=pbar, **kwargs)

            if insert_stats:
                with pbar_update(pbar, desc='Inserting statements...') as p: 
                    self.insert_project_statements(pbar=pbar, **kwargs)

            with pbar_update(pbar, desc='Update formats...') as p: 
                self.update_formats(pbar=pbar, **kwargs)

            pbar.set_description('Project Sim Complete.')

        if complete_msg:
            msgbox(self.state.wb, 'Project Simulation Complete.')

    def generate_project_statements(self, pbar=None, **kwargs):
        projstats = ProjectStats(self.state.projects, self.state.periods, pbar=pbar)
        projstats.finalize(self.state.env, pbar=pbar) # .env added to state in super() call above
        self.state.set_project_statements(projstats)

    def insert_project_statements(self, pbar, **kwargs):
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

    def create_project_charts(self, **kwargs):
        pbar = kwargs.pop('pbar', None)

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

    def get_environments_to_load(self):
        profiles_to_load('envs', self.state)

    def get_miner_profiles_to_load(self):
        profiles_to_load('miners', self.state)

    def get_pool_profiles_to_load(self):
        profiles_to_load('pools', self.state)

    def get_project_profiles_to_load(self):
        profiles_to_load('projects', self.state)

    def load_miners(self):
        load_profiles('miners', self.state)

    def load_pools(self):
        load_profiles('pools', self.state)

    def load_projects(self):
        load_profiles('projects', self.state)

    def load_environment(self):
        sheet_name=self.state.WS['envs'] 
        loadcell = self.state.LOAD_CELLS['envs']
        
        dirname = self.state.wb.sheets[sheet_name].range(loadcell).value
        self.state.load_environment(dirname)
        print (f'Environment has been updated.')

    def save_environment(self):
        dirname = self.state.wb.app.api.InputBox('Input Folder Name', '', Type=2)
        
        if dirname:
            self.state.save_environment(dirname)

    def warn_env(self):
        ws = self.state.wb.sheets[self.state.WS['meta']]
        btn_warning(ws, self.state.BUTTONS['env'], 'Simulate Environment')

    def warn_projs(self):
        ws = self.state.wb.sheets[self.state.WS['meta']]
        btn_warning(ws, self.state.BUTTONS['proj'], 'Simulate Projects')

    def warn(self, model_type):
        ws = self.state.wb.sheets[self.state.WS[model_type]]
        btn_warning(ws, self.state.BUTTONS[model_type], 'Update')
        self.state.wb.sheets(self.state.WS['meta']).range(self.state.CHECKLIST_CELLS[model_type]).value = 0

    def warn_sched(self):
        self.warn('block_sched')

    def warn_btc(self):
        self.warn('btc_price')

    def warn_fees(self):
        self.warn('fees')

    def update_button(self, *args, **kwargs):
        button_update(self.state, *args, **kwargs)

    def pass_checklist(self):
        pass_checklist(self.state)

    def break_checklist(self):
        self.warn_projs()
        self.warn_env()
        self.warn_sched()
        self.warn_btc()
        self.warn_fees()

def on_import(
    ModelMaker, 
    filepath, 
    load_profile_dropdowns=True,
    state_constructor=None,
    complete_msg=True,
    update_mines=False,
    ):
    with tqdm(total=5) as t:
        with pbar_update(t, desc='Initiaiting state...') as p:
            if state_constructor is None:
                from .state import SheetState
                state_constructor = SheetState
            State = state_constructor(filepath)
            State.update_btc()
            modelmkr = ModelMaker(State)

        if load_profile_dropdowns:
            modelmkr.get_environments_to_load()
            modelmkr.get_miner_profiles_to_load()
            modelmkr.get_pool_profiles_to_load()
            modelmkr.get_project_profiles_to_load()

        if not State.LOAD_ON_IMPORT:
            modelmkr.break_checklist()

            if complete_msg:
               msgbox(State.wb, 'Import complete.\nYOU MUST LOAD AN ENVIRONMENT.')

            t.update(5)
        else:
            State.wb.app.enable_events = False # Allows spreadsheet changes with triggering event handlers
            with pbar_update(t, desc='Updating profiles...') as p:
                State.update_miners()
                State.update_cooling()
                if update_mines:
                    State.update_pools()
                    State.update_projects()

                with pbar_update(t, desc='Generating block schedule...') as p:
                    modelmkr.generate_block_schedule()

                with pbar_update(t, desc='Generating BTC price forecast...') as p:
                    modelmkr.generate_btc_forecast()

                with pbar_update(t, desc='Generating transaction fee forecast...') as p:
                    modelmkr.generate_fee_forecast()

            modelmkr.warn_env()
            modelmkr.warn_projs()
            State.wb.app.enable_events = True

            if complete_msg:
                msgbox(State.wb, 'Import complete.')

    t.set_description('Complete.')

    return State, modelmkr
