import numpy as np
import pandas as pd

from tqdm.auto import trange

from .funcs import cooling_energy

from numpy.random import default_rng
np_rand = default_rng()

ARRAY_KEY = {k: v for k, v in zip(['hashes', 'miner_energy', 'revenues', 'cogs'], np.arange(4))}

def retgt_ranges(sched, retarget_blocks):
    blocks_to_first_retgt = ((sched[sched.retarget].index[0] - sched.index[0]).n // 10)
    blocks_from_last_retgt = ((sched.index[-1] - sched[sched.retarget].index[-1]).n // 10) + 1
    lasts = np.arange(blocks_to_first_retgt, sched.index.size - blocks_from_last_retgt + retarget_blocks, retarget_blocks)
    lasts = np.concatenate((lasts, np.array([lasts[-1] + blocks_from_last_retgt])))
    firsts = np.roll(lasts, 1)
    firsts[0] = 0
    return np.vstack((firsts, lasts)).T

def choice2d(a, axis=0):
    # https://stackoverflow.com/questions/47722005/vectorizing-numpy-random-choice-for-given-2d-array-of-probabilities-along-an-a
    r = np.expand_dims(np_rand.random(a.shape[1-axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)

def sim_mining(
    static_hashes_per_miner, 
    static_energy_per_miner, 
    energy_cost,
    pues,
    ends, 
    block_revenues,
    n_miners, 
    halted
    ):
    n_blocks = np.diff(ends)[0]
    block_revenues = block_revenues[slice(*ends)]
    n_miners = n_miners[:, slice(*ends)]
    
    hashes_per_miner = static_hashes_per_miner * ~halted
    energy_per_miner = static_energy_per_miner * ~halted

    miner_energy = (n_miners.T * energy_per_miner).T
    cool_energy = np.array([cooling_energy(e, pue) for e, pue in zip(miner_energy, pues)])
    energy = miner_energy + cool_energy
    
    hashes = (n_miners.T * hashes_per_miner).T
    shares = hashes / hashes.sum(axis=0)
    
    assert np.allclose(shares.sum(axis=0), 1), shares.sum(axis=0)
    winners = choice2d(shares)

    arr = np.zeros(shares.shape, dtype='int')*np.nan
    arr[0, :] = winners
    rows, cols = np.ogrid[:arr.shape[0], :arr.shape[1]]
    rows = rows - winners[np.newaxis, :]
    arr = arr[rows, cols]

    revenues = ~np.isnan(arr) * block_revenues
    cogs = (energy.T * energy_cost).T

    return np.hstack((hashes, miner_energy, revenues, cogs)).reshape(pues.size, -1, n_blocks)

def assess(sched, miner_scheds, financials, rng, min_gm=0):
    end = rng[-1]
    assessment_period = 90
    
    check_start = sched.index[end - 1] - pd.tseries.offsets.Day(assessment_period)

    if check_start in sched.index:
        periods = (sched.index[end - 1] - check_start).n // 10
        
        rev_for_period = financials[:, -2, -periods:].sum(axis=1)
        cogs_for_period = financials[:, -1, -periods:].sum(axis=1)
        gp = rev_for_period - cogs_for_period
        
        with np.errstate(divide='ignore', invalid='ignore'):
            gm = gp / rev_for_period        
            gm = np.where(np.isnan(gm), min_gm, gm)
            
        launched = (miner_scheds[:, slice(*rng)] > 0).all(axis=1)        
        halted = (gm <= min_gm) & launched
    else:
        halted = np.zeros(financials.shape[0], dtype='bool')
        
    return halted
        
def simulate(mining, sched, revenues, ends, pbar_kws={}):
    miner_scheds = np.vstack([m.implement(sched.index.size).miner_schedule for m in mining])
    hashes_per_miner = np.array([m.miner.hr.hashes_per_block() for m in mining])
    energy_per_miner = np.array([m.miner.power.consumption_per_block() for m in mining])
    energy_cost = np.array([m.energy_cost for m in mining])
    pues = np.array([m.cooling.pue for m in mining])

    tot_fins = []
    halted = np.zeros(mining.size, dtype=bool)
    halts = [halted]
    for i in trange(ends.shape[0], **pbar_kws):
        financials = sim_mining(hashes_per_miner, energy_per_miner, energy_cost, pues, ends[i], revenues, miner_scheds, halted)
        tot_fins.append(financials)

        halted = assess(sched, miner_scheds, np.dstack(tot_fins), ends[i], min_gm=.2)
        if i < ends.shape[0] - 1:
            halts.append(halted)

    tot_fins = np.dstack(tot_fins)
    halts = np.stack(halts).T
    
    return tot_fins, halts

