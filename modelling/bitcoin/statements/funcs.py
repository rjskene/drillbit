import numpy as np
import finstat as fs
from btc.funcs import mining_rate

@fs.statfunc
def expected_difficulty(hashes):
    from btc.funcs import expected_difficulty
    return expected_difficulty(hashes)

@fs.statfunc
def win_percentage(hashes, difficulty):
    return hashes / (2**32 * difficulty)

@fs.statfunc
def hashes_to_hash_rate(hashes):
    return hashes / (10 * 60)

@fs.statfunc
def btc_mined(difficulty, hash_rate, rewards, periods):
    return np.array([mining_rate(diff, hash_rate, reward, days) for diff, reward, days in np.vstack((difficulty, rewards, periods.days_in_month)).T])

@fs.statfunc
def cooling_energy(energy, pue):
    return energy * (pue - 1)

@fs.statfunc
def amort_sched(unit_cost, units, n_amort, start_month, periods, days_in_month=30):
    n_amort = int(n_amort * (days_in_month * 24 * 6))
    amort = np.zeros(periods.size)
    amort[start_month : n_amort + start_month] = unit_cost * units / n_amort
    return amort

@fs.statfunc
def miner_amort(profile, periods):
    new_miners = np.zeros(periods.size)
    new_miners[0] = profile.miner_schedule[0]
    new_miners[1:] = profile.miner_schedule[1:] - profile.miner_schedule[:-1]
    new_miners *= profile.miner.price

    n_amort = int(profile.miner.amort * (30 * 24 * 6))
    return np.cumsum(new_miners / n_amort)