import datetime as dt
import numpy as np
import pandas as pd

from .units import Hashes, HashRate, Energy

DIFFICULTY_1_hash = 0x00ffff * 2**(8*(0x1d - 3))

def consumption_in_Wh(power, **duration):
    if not duration:
        duration = {'days': 30}
    
    assert len(duration) <= 1, 'Only one set of duration inputs currently supported'

    for key, value in duration.items():
        try:
            iterator = iter(value)
        except TypeError:
            duration_in_hours = pd.Timedelta(**duration) / pd.Timedelta(hours=1)
        else:
            duration_in_hours = pd.to_timedelta(value, unit=key) / pd.Timedelta(hours=1)

    return Energy(power * duration_in_hours, abbr='Wh')

def decompress(bits):
    """
    Convert compressed integer version of target found in 'bits' field
    https://medium.com/fcats-blockchain-incubator/understanding-the-bitcoin-blockchain-header-a2b0db06b515
    """
    exp = bits[:2]
    mantissa = bits[2:]
    target = int(mantissa, 16) * (2**(8*(int(exp, 16) - 3)))
    return target

def time_to_block(difficulty, hash_rate):
    return difficulty * 2**32 / hash_rate

def expected_hashes(difficulty):
    return difficulty * 2**256 / (0xffff * 2**208)

def expected_difficulty(hashes):
    return hashes * DIFFICULTY_1_hash  / 2**256

def calc_difficulty(tgt):
    return DIFFICULTY_1_hash / tgt

def calc_target(difficulty):
    return DIFFICULTY_1_hash / difficulty

def calc_network_hash_rate(difficulty):
    return HashRate(expected_hashes(difficulty) / 600)

def solve_likelihood_per_hash(difficulty):
    return 1/(2**32 * difficulty)

def solve_likelihood_per_sec(hash_rate, difficulty):
    return hash_rate / (difficulty * (2**32))

def mining_rate(difficulty, hash_rate, reward, days=1):
    ### Should be rearranged; not intuitive; Difficulty is per block; 
    ### so should solve hashes per block, then number of blocks across n days
    ### so, reward_per_block = hashes_per_block * reward * likelihood of solving per hash; reward_per_block * 6 * 24 * n days
    hashes_per_block = Hashes(hash_rate * dt.timedelta(minutes=10).total_seconds())
    reward_per_block = reward * hashes_per_block * solve_likelihood_per_hash(difficulty)
    avg_reward = reward_per_block * 6 * 24 * days
    return avg_reward

def cooling_energy(energy, pue):
    return energy * (pue - 1)

def compound_growth(init, g, n):
    arr = np.ones(n)
    arr[1:] += g
    return init * arr.cumprod()

def geometric_brownian_motion(S0, n, T, mu=0.01, sigma=0.01):
    dt = T / n
    t = np.arange(n)
    W = np.random.standard_normal(size=n)
    W = np.cumsum(W)*np.sqrt(dt) ### standard brownian motion ###
    X = (mu - 0.5*sigma**2)*t + sigma*W
    return S0*np.exp(X)

class MiningMixin:
    DIFFICULTY_1_hash = DIFFICULTY_1_hash

    @property
    def DIFF1(self):
        return self.DIFFICULTY_1_hash

    def difficulty_from_tgt(self, target=None):
        target = self._TARGET_HASH if target is None else target
        return calc_difficulty(target)

    def tgt_from_difficulty(self, difficulty):
        return calc_target(difficulty)

    def difficulty_from_hashes(self, hashes):
        return expected_difficulty(hashes)

    def expected_hash_rate(self, difficulty=None):
        difficulty = self.difficulty_from_tgt() if difficulty is None else difficulty
        return calc_network_hash_rate(difficulty)

    def expected_hashes(self):
        return expected_hashes(self.difficulty_from_tgt())

    def solve_likelihood_per_hash(self, difficulty=None):
        difficulty = self.difficulty if difficulty is None else difficulty
        return solve_likelihood_per_hash(difficulty)

    def likelihood_per_block(self, hash_rate, difficulty=None, likelihood_per_hash=None):
        if likelihood_per_hash is None:
            likelihood_per_hash = self.solve_likelihood_per_hash(difficulty)

        return hash_rate.hashes_per_block() * likelihood_per_hash

    def revenue_per_block(self, price, reward, traxn_fee, likelihood=1):
        reward_revenue = reward * price
        fee_revenue = traxn_fee * price
        
        return likelihood * (reward_revenue + fee_revenue)

    def cogs_per_block(self, power, cost):
        return power.consumption_per_block() * cost

    def cgr(self, price, n, g):
        return compound_growth(price, g, n)

    def gbm(self, price, periods, *args, **kwargs):
        n = periods.size
        T = (periods[-1] - periods[0]).delta.days
        return geometric_brownian_motion(price, n, T, *args, **kwargs)
