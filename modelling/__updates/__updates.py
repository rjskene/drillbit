import xlwings as xw

from drillbit.price_lookups import price_lookup
from drillbit.demo import mining_demo, mining_demo_avg
# from drillbit.state import SheetState
from .excel import GenericModelMaker, on_import

@xw.func
def tbl_rng(ws, tbl):
    # Used for Worksheet_change calls in VBA to trigger State updates
    return State.get_table(ws, tbl).address

@xw.func
def update_config():
    State.update_config()
    modelmkr.warn_env()

@xw.func
def update_miners():
    State.update_miners()
    # State.update_mines() # have to update mines for changes in cooling objects
    modelmkr.warn_env()

@xw.func
def update_cooling():
    State.update_cooling()
    # State.update_mines() # have to update mines for changes in cooling objects
    modelmkr.warn_env()

@xw.func
def update_pools():
    # State.update_mines()
    modelmkr.warn_env()

@xw.func
def update_projects():
    # State.update_mines()
    modelmkr.warn_projects()

@xw.func
def warn_btc():
    modelmkr.warn_btc()

@xw.func
def warn_fees():
    modelmkr.warn_fees()

@xw.func
def break_checklist():
    modelmkr.break_checklist()

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
def load_environment():
    modelmkr.load_environment()
    modelmkr.update_button(State.get_ws('meta'), 'env', False)
    modelmkr.pass_checklist()

@xw.sub
def save_environment():
    modelmkr.save_environment()

@xw.sub
def load_miners():
    modelmkr.load_miners()

@xw.sub
def load_pools():
    modelmkr.load_pools()

@xw.sub
def load_projects():
    modelmkr.load_projects()

@xw.sub
def simulate_mining_env():
    State.update_pools()
    State.implement_pools()

    if not State.ENV_ONLY:
        State.update_projects()
        State.implement_projects()
    
    modelmkr.create_environment(
        btn_sheet='meta',
        btn_name=State.BUTTONS['env'],
        tracker_sheet='meta',
        tracker_cells=list(State.CHECKLIST_CELLS.values()),
        chart_sheet='minenv',
        pbar_kws=dict(total=140),
        wipe_stats=True
    )

@xw.sub
def simulate_projects():
    State.update_projects()
    State.implement_projects()

    modelmkr.create_projects(
        btn_sheet='meta',
        btn_name=State.BUTTONS['proj'],
        pbar_kws=dict(total=32)
    )

### Start-up code ONLY executes if there is a recognizable
### event loop running 
### this should only happen on "Import functions" call from inside Excel
### and not when import elsewhere like Jupyter
from xlwings.server import loop
if loop.is_running() and __name__.split('.')[0] in xw.books.active.name:
    State, modelmkr = on_import(GenericModelMaker, __file__)

"""
FUNCTIONS USED FOR TESTING
"""
# @xw.func
# def miner_eff(mine):
#     return State.mines[mine].miner.eff
