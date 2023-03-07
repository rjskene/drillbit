"""
See https://en.bitcoin.it/wiki/Difficulty
"""
import typing as typ
from nptyping import NDArray, Int, Shape

import numpy as np
import finstat as fs

from ..mining import mining_rate

@fs.statfunc
def expected_difficulty(hashes):
    from ..mining import expected_difficulty
    return expected_difficulty(hashes)

@fs.statfunc
def win_percentage(hashes, difficulty):
    return hashes / (2**32 * difficulty)

@fs.statfunc
def hashes_to_hash_rate(hashes):
    return hashes / (10 * 60)

@fs.statfunc
def hash_rate_to_hashes(hash_rate):
    return hash_rate * (10 * 60)
 
@fs.statfunc
def btc_mined(difficulty, hash_rate, rewards, periods):
    return np.array([mining_rate(diff, hash_rate, reward, days) for diff, reward, days in np.vstack((difficulty, rewards, periods.days_in_month)).T])

@fs.statfunc
def total_energy(energy, pue):
    return energy * (pue - 1)

@fs.statfunc
def amortize(n, n_amort, price, quantity, start_month=0):
    n_amort = int(n_amort * (30 * 24 * 6))
    amort = np.zeros(n)
    amort[start_month : n_amort + start_month] = price * quantity / n_amort
    return amort

@fs.statfunc
def staggered_amortize(n, n_amort, price, quantities):
    """
    Amortize the cost of an asset over a given number of periods. The implementation of the
    asset is staggered over the given periods, per the quantities array.

    Parameters
    ----------
    n : int
        number of periods in the statemet
    price : float
        price of the miner
    n_amort : int
        number of months to amortize the miner
    quantity : NDArray
        quantity of miners deployed each period

    Returns
    -------
    NDArray of amortization schedule
    """
    n_amort = int(n_amort * (30 * 24 * 6)) # convert months to periods of 10 minutes
    quant_arr = np.zeros(n)
    quant_arr[0] = quantities[0]
    # find the number of new quantity deployed each period
    # must use `.values` to get the underlying numpy array, b/c pandas does not support negative indexing
    quant_arr[1:] = quantities[1:].values - quantities[:-1].values 

    non_zero_indices = np.nonzero(quant_arr)[0]

    # create an empty array to hold the amortized values
    amortized_arr = np.zeros(len(quant_arr))

    # amortize each non-zero value over a the given periods
    for i in non_zero_indices:
        start_index = i
        end_index = min(i + n_amort, len(quant_arr))
        amortized_value = price * quant_arr[i] / (end_index - start_index)
        amortized_arr[start_index:end_index] += amortized_value

    return amortized_arr
