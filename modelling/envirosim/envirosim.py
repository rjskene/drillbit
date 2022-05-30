from tqdm.auto import tqdm
import xlwings as xw

from bitcoin.price_lookups import price_lookup
from .excel import EnviroModelMaker, on_import

@xw.func
def update_miners():
    State.update_miners()
    modelmkr.warn_sim()

@xw.func
def update_cooling():
    State.update_cooling()
    modelmkr.warn_sim()

@xw.func
def update_mines():
    State.update_mines()
    modelmkr.warn_sim()

@xw.func
def warn_btc():
    modelmkr.warn_btc()

@xw.func
def warn_fees():
    modelmkr.warn_fees()

@xw.func
def break_checklist():
    modelmkr.warn_sim()
    modelmkr.warn_sched()
    modelmkr.warn_btc()
    modelmkr.warn_fees()

@xw.sub
def update_meta():
    State.update_btc()

@xw.sub
def generate_block_schedule():
    modelmkr.generate_block_schedule()

@xw.sub
def generate_btc_forecast():
    modelmkr.generate_btc_forecast()

@xw.sub
def generate_fee_forecast():
    modelmkr.generate_fee_forecast()

@xw.sub
def load_miners():
    modelmkr.load_miners()

@xw.sub
def load_mines():
    modelmkr.load_mines()

@xw.sub
def simulate_mining_env():
    modelmkr.create(
        btn_sheet=State.WS['meta'],
        btn_name=State.BUTTONS['sim'],
        tracker_sheet=State.WS['meta'],
        tracker_cells=list(State.CHECKLIST_CELLS.values()),
        chart_sheet=State.CHARTS['env'],
        pbar_kws=dict(total=12)
    )

### Start-up code ONLY executes if there is a recognizable
### event loop running 
### this should only happen on "Import functions" call from inside Excel
### and not when import elsewhere like Jupyter
from xlwings.server import loop
if loop.is_running() and __name__.split('.')[0] in xw.books.active.name:
    State, modelmkr = on_import(EnviroModelMaker, __file__)
