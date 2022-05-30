from tqdm.auto import tqdm
from xlwings.utils import rgb_to_int

from .state import SheetState
from .charts import charts
from .style import grc_style

from abc import ABC, abstractmethod

def btn_warning(ws, btn_name, msg):
    btn = ws.api.OLEObjects(btn_name)
    btn.Object.Caption = msg
    btn.Object.BackColor = rgb_to_int((255,0,0))

def button_update(state, ws, model):
    ws.api.OLEObjects(state.BUTTONS[model]).Object.BackColor = rgb_to_int(grc_style.vvlblue.hex_to_rgb())
    ws.api.OLEObjects(state.BUTTONS[model]).Object.Caption = 'Updated!'
    state.wb.sheets[state.WS['meta']].range(state.CHECKLIST_CELLS[model]).value = 1

class AbstractBaseModelMaker(ABC):
    def __init__(self, state):
        self.state = state

    @abstractmethod
    def validation(self, *args, **kwargs):
        pass

    @abstractmethod
    def initialize(self, **kwargs):
        pass

    @abstractmethod
    def create_charts(self, pbar, **kwargs):
        pass

    @abstractmethod
    def insert_statements(self, **kwargs):
        pass

    @abstractmethod
    def update_formats(self, **kwargs):
        pass

    @abstractmethod
    def create(self, **kwargs):
        pass

    def generate_block_schedule(self):
        ws = self.state.wb.sheets[self.state.WS['block_sched']]   
        sched = self.state.generate_block_schedule()

        ws.range(self.state.TABLES['block_sched']).options(index=True, header=True).value = sched.iloc[:100]
        button_update(self.state, ws, 'block_sched')
        charts.chart_block_sched(self.state, ws)

    def generate_btc_forecast(self):
        ws = self.state.wb.sheets[self.state.WS['btc_price']]
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

    def warn_sim(self):
        ws = self.state.wb.sheets[self.state.WS['meta']]
        btn_warning(ws, self.state.BUTTONS['sim'], 'Simulate')

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