import pandas as pd

from tqdm.auto import tqdm
from xlwings.utils import rgb_to_int

from btc import sim

from bitcoin.charts import charts
from bitcoin.style import grc_style
from bitcoin.state import SheetState
from bitcoin.helpers import pbar_update
from bitcoin.excel import AbstractBaseModelMaker, msgbox

from .statements import MineStats, init_enviro
from .charts import chart_operational_mines, count_operational, mine_gross_margin, env_sim_pre_processing, mine_hash_rate

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
    profiles = [file_.name.split('.')[0].title() for file_ in datadir.iterdir() if '~' not in file_.name]
    State.wb.sheets[sheet_name].range(loadcell).api.Validation.Delete()
    State.wb.sheets[sheet_name].range(loadcell).api.Validation.Add(Type=3,Formula1=','.join(profiles))

def load_profiles(profile_type, State):
    sheet_name=State.WS[profile_type] 
    table_name=State.TABLES[profile_type] 
    loadcell=State.LOAD_CELLS[profile_type]
    
    ws = State.wb.sheets[sheet_name]
    filename = ws.range(loadcell).value
    mine_profiles = State.parent_path.parent / 'data' / profile_type / f'{filename.lower()}.xlsx'
    df = pd.read_excel(mine_profiles)

    State.wb.app.enable_events = False
    ws.range(f'{table_name}').clear_contents()
    State.wb.app.enable_events = True
    ws.range(f'{table_name}').options(pd.DataFrame, header=False, index=False).value = df
    print (f'{profile_type.title()} update.')

class EnviroModelMaker(AbstractBaseModelMaker):
    def validation(self, *args, **kwargs):
        return check_ready(self.state)

    def simulate_mining_environment(self, env, mines):
        ends = sim.retgt_ranges(self.state.block_sched, self.state.BTC.RETARGET_BLOCKS)
        financials, _ = sim.simulate(mines, self.state.block_sched, env.mkt_rev.values, ends, pbar_kws={'leave': False})
        return financials

    def generate_statements(self, mines, pbar=None, **kwargs):
        financials = kwargs.pop('financials')
        env = kwargs.pop('env')

        self.state.mines.implement(self.state.periods.size)
        minerstats = MineStats(mines, financials[:, sim.ARRAY_KEY['miner_energy']], financials[:, sim.ARRAY_KEY['hashes']], self.state.periods, pbar=pbar)
        minerstats.finalize(env, financials[:, sim.ARRAY_KEY['hashes']].sum(axis=0), pbar=pbar)
        self.state.set_global_environment(env)
        self.state.set_mine_statements(minerstats)

    def initialize(self, pbar=None, **kwargs):
        env = init_enviro(self.state.block_sched, self.state.traxn_fees, self.state.btc_price)

        with pbar_update(pbar, desc='Creating environment data...') as p:
            if self.state.pools_only: # whether to include only Pools or both Pools & Projects in environment
                mines = self.state.mines.pools
            else:
                mines = self.state.mines

            financials = self.simulate_mining_environment(env, mines)

        with pbar_update(pbar, desc='Generating mining statements...') as p:
            self.generate_statements(mines, financials=financials, env=env, pbar=pbar)

    def create_charts(self, **kwargs):
        pbar = kwargs.pop('pbar', None)
        chart_sheet = kwargs.pop('chart_sheet')
        op, count, gm, halve_ticks, hr = env_sim_pre_processing(self.state)

        chart_operational_mines(op, halve_ticks, chart_sheet, self.state.parent_path)
        if pbar is not None:
            pbar.update(1)

        mine_gross_margin(gm, halve_ticks, chart_sheet, self.state.parent_path)
        if pbar is not None:
            pbar.update(1)

        mine_hash_rate(hr, halve_ticks, chart_sheet, self.state.parent_path)
        if pbar is not None:
            pbar.update(1)

        count_operational(count, chart_sheet, self.state.parent_path)
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

    def insert_statements(self, **kwargs):
        chart_sheet = kwargs.pop('chart_sheet')
        chart_sheet.range('A1').options(header=True, index=True).value = self.state.env.to_frame(with_periods=False).T.iloc[:100]

    def update_formats(self, **kwargs):
        btn_sheet = kwargs.pop('btn_sheet')
        btn_name = kwargs.pop('btn_name')

        btn_sheet.api.OLEObjects(btn_name).Object.BackColor = rgb_to_int(grc_style.vvlblue.hex_to_rgb())
        btn_sheet.api.OLEObjects(btn_name).Object.Caption = 'Updated!'

    def create(self, complete_msg=True, pbar_kws={}, **kwargs):
        update_charts = kwargs.pop('update_charts', self.state.update_charts)
        insert_stats = kwargs.pop('insert_stats', self.state.insert_statements)
        wipe_stats = kwargs.pop('wipe_stats', False)
        if wipe_stats:
            self.state.wipe()

        kwargs['btn_sheet'] = self.state.wb.sheets[kwargs.get('btn_sheet', 'BTC Meta')]
        kwargs['tracker_sheet'] = self.state.wb.sheets[kwargs.get('tracker_sheet', 'BTC Meta')]
        kwargs['chart_sheet'] = self.state.wb.sheets[kwargs.get('chart_sheet', 'Mining Environment')]

        validated = self.validation(**kwargs)
        if validated:
            with tqdm(**pbar_kws) as pbar:
                pbar.set_description('Initializing...')
                self.initialize(pbar=pbar, **kwargs)

                if update_charts:
                    pbar.set_description('Creating charts...')
                    self.create_charts(pbar=pbar, **kwargs)

                if insert_stats:
                    with pbar_update(pbar, desc='Inserting statements...') as p: 
                        self.insert_statements(pbar=pbar, **kwargs)

                with pbar_update(pbar, desc='Update formats...') as p: 
                    self.update_formats(pbar=pbar, **kwargs)
   
                pbar.set_description('Complete.')

        if complete_msg:
            msgbox(self.state.wb, 'Simulation Complete.')

    def get_miner_profiles_to_load(self):
        profiles_to_load('miners', self.state)

    def get_pool_profiles_to_load(self):
        profiles_to_load('pools', self.state)

    def get_mine_profiles_to_load(self):
        profiles_to_load('mines', self.state)

    def load_miners(self):
        load_profiles('miners', self.state)

    def load_mines(self):
        load_profiles('mines', self.state)

    def load_pools(self):
        load_profiles('pools', self.state)

def on_import(ModelMaker, filepath, load_profiles=True, state_constructor=None, complete_msg=True):
    with tqdm(total=5) as t:
        with pbar_update(t, desc='Initiaiting state...') as p:
            if state_constructor is None:
                state_constructor = SheetState
            State = state_constructor(filepath)
            State.update_btc()
            modelmkr = ModelMaker(State)

        State.wb.app.enable_events = False
        with pbar_update(t, desc='Updating profiles...') as p:
            State.update_miners()
            State.update_cooling()
            State.update_mines()

            if load_profiles:
                modelmkr.get_miner_profiles_to_load()
                modelmkr.get_pool_profiles_to_load()

        with pbar_update(t, desc='Generating block schedule...') as p:
            modelmkr.generate_block_schedule()

        with pbar_update(t, desc='Generating BTC price forecast...') as p:
            modelmkr.generate_btc_forecast()

        with pbar_update(t, desc='Generating transaction fee forecast...') as p:
            modelmkr.generate_fee_forecast()

        State.mines.implement(State.periods.size)
        modelmkr.warn_sim()
        State.wb.app.enable_events = True

    if complete_msg:
        msgbox(State.wb, 'Import complete.')

    return State, modelmkr
