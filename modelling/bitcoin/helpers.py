import string
import itertools as it
import win32api
import win32con

from xlwings.conversion import Converter, PandasDataFrameConverter

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

class DataFrameDropna(Converter):

    base = PandasDataFrameConverter

    @staticmethod
    def read_value(builtin_df, options):
        return builtin_df.dropna()

class pbar_update:
    def __init__(self, pbar, desc='', update=True):
        self.pbar = pbar
        self.desc = desc
        self.update = update

    def __enter__(self):
        if self.pbar is not None:
            self.pbar.set_description(self.desc)

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type:
            raise exc_type(exc_value).with_traceback(exc_tb)
        
        if self.pbar is not None and self.update:
            self.pbar.update(1)
