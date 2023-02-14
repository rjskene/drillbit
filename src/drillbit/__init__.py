"""
### TO DO ###
Need to improve Units module; units need to adapt to scale on the fly
Need internal representation of actual and most concise units
"""

from .environment import BitcoinEnvironmentUtility
from .objects import Miner, Miners, Cooling, CoolingProfiles, Mining, MiningProfiles
from .units import *