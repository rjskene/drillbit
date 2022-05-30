from collections import UserString
import numpy as np

import matplotlib.ticker as mticks
import matplotlib as mpl
from matplotlib import cm
from cycler import cycler

class ColorStr(UserString):
    @property
    def nohash(self):
        return self.data[1:]
    
    @property
    def hash(self):
        return str(self.data)

    def hex_to_rgb(self, as_array=False, normed=False): 
        rgb = tuple(int(self.nohash[i:i+2], 16) for i in (0, 2, 4))
        if as_array:
            rgb = np.array(rgb)
            if normed:
                rgb = rgb / 255

        return rgb

    def rgb_to_bgrhex(self, rgb):
        '''
        win32 uses bgr in hex
        '''
        bgr = (rgb[2], rgb[1], rgb[0])
        strValue = '%02x%02x%02x' % bgr
        iValue = int(strValue, 16)

        return iValue

    @property
    def bgrhex(self):
        if not hasattr(self, '_bgrhex'):
            self._bgrhex = self.rgb_to_bgrhex(self.hex_to_rgb())
        
        return self._bgrhex

class GRCStyle:
    green = ColorStr('#6dbc45')
    lgreen = ColorStr('#b8dfa5')
    vlgreen = ColorStr('#D0EAC3')
    vvlgreen = ColorStr('#E7F4E1')
    vvvlgreen = ColorStr('#f3f9f0')
    blue = ColorStr('#253765')
    lblue = ColorStr('#2765b0')
    vlblue = ColorStr('#c5d9f3')
    vvlblue = ColorStr('#eef4fb')

    @property
    def _colors(self):
        return [self.blue, self.lblue, self.vlblue, self.vvlblue, self.green, self.lgreen, self.vlgreen, self.vvlgreen]

    def _color_array(self, lcolor, rcolor, size=256, normed=True, with_alpha=False):
        arr = np.linspace(
            lcolor.hex_to_rgb(as_array=True, normed=normed), 
            rcolor.hex_to_rgb(as_array=True, normed=normed), 
            size
        )
        if with_alpha:
            arr = np.hstack((arr, np.ones((arr.shape[0], 1))))

        return arr

    def blues(self, *args, **kwargs):
        return self._color_array(self.vvlblue, self.blue, *args, **kwargs)

    def greens(self, *args, **kwargs):
        return self._color_array(self.vvvlgreen, self.green, *args, **kwargs)

    def reds(self, size=256):
        return cm.get_cmap('Reds_r', size)(np.linspace(0, 1, size))

    def colors(self, nohash=True, alternate=False, rgb=False):
        colors = self._colors

        if rgb:
            colors = [c.hex_to_rgb() for c in colors]
        elif nohash:
            colors = [c.nohash for c in colors]
        else:
            colors = [c.hash for c in colors]

        if alternate:
            colors = [c for zips in zip(colors[:3], colors[4:-1]) for c in zips]        

        return colors

    @staticmethod
    @mticks.FuncFormatter
    def dol_fmt(x, pos):
        return f'${x:,.0f}'

    @staticmethod
    @mticks.FuncFormatter
    def mill_fmt(x, pos):
        return f'${x / 1e6:,.0f}M'

    @staticmethod
    @mticks.FuncFormatter
    def btc_fmt(x, pos):
        return f'\u0243{x:,.3f}'.rstrip('0')

    @staticmethod
    @mticks.FuncFormatter
    def diff_fmt(x, pos):
        return f'{x / 10**12:,.0f}T'

    @staticmethod
    @mticks.FuncFormatter
    def hr_fmt(x, pos):
        return f'{x / 10**6:,.0f}M TH/s'

    @staticmethod
    @mticks.FuncFormatter
    def per_fmt(x, pos):
        return f'{x:.1%}'.replace('.0', '')
        
    @staticmethod
    @mticks.FuncFormatter
    def pow_fmt(x, pos):
        return f'{x / 10**3:,.1f} kW'

    def set_rc(self):
        mpl.rcParams['figure.figsize'] = 16, 4
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.right'] = False

        mpl.rcParams['axes.prop_cycle'] = cycler('color', self.colors(nohash=False, alternate=True))

        mpl.rcParams['axes.titlecolor'] = self.blue.nohash
        mpl.rcParams['axes.titlesize'] = 24
        mpl.rcParams['figure.titlesize'] = 24

        mpl.rcParams['axes.labelcolor'] = self.blue.nohash

grc_style = GRCStyle()
grc_style.set_rc()