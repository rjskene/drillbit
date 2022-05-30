import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from bitcoin.style import grc_style
from bitcoin.charts.charts import render, PLOT_KWARGS

def env_sim_pre_processing(State):
    rev = State.minestats.by_lineitem('rev', how='frame').T.resample('2W').sum().T
    cogs = State.minestats.by_lineitem('energy_exp', how='frame').T.resample('2W').sum().T
    gp = State.minestats.by_lineitem('gp', how='frame').T.resample('2W').sum().T
    gm = gp / rev
    operating = ~(cogs == 0)

    def series_and(ser):
        return ser.any(0) 

    op = operating.T.resample('2W').apply(series_and)
    count = op.sum(axis=1)

    halvings = State.block_sched.reward[State.block_sched.reward.shift(1) != State.block_sched.reward].iloc[1:]
    halve_ticks = op.index.get_indexer(halvings.index.asfreq('2W'), method='nearest')

    return op, count, gm, halve_ticks

def chart_operational_mines(op, halves, ws, parent_path):
    fig, ax = plt.subplots(figsize=(16,8), **{k:v for k, v in PLOT_KWARGS.items() if k not in ['figsize']})

    grcmap = ListedColormap([(1,1,1,1), grc_style.blue.hex_to_rgb(as_array=True, normed=True)])
    im = ax.imshow(op.T, cmap=grcmap, aspect='auto')

    xticks = ax.get_xticks().astype(int)
    yticks = np.arange(op.shape[1])
    ax.set_xticks(xticks[1:-1], labels=op.index.strftime('%Y-%b')[xticks[1:-1]])
    ax.set_yticks(yticks, minor=False, labels=op.columns)
    ax.set_yticks(yticks + .5, minor=True)

    halve_lines = ax.vlines(halves, yticks[0], yticks[-1], colors=grc_style.green.hash, ls='--', lw=2, label='Halving')

    ax.tick_params(axis='y', which='both', width=0)
    ax.grid(axis='y', which="minor", color="w", linestyle='-', linewidth=2)

    ax.spines[:].set_visible(False)
    ax.legend(loc='best', framealpha=0.5)
    ax.set_title('Mine Operation')

    render(ws, fig, parent_path, 'fees', ws.name + ' Historical Fee')

def count_operational(count, ws, parent_path):
    fig, ax = plt.subplots(**PLOT_KWARGS)

    count.plot(ax=ax)
    ax.set_xlabel('')
    ax.set_title('Number of Operational Mines')

    render(ws, fig, parent_path, 'count_mines', ws.name + ' Count of Operational Mines')

def mine_gross_margin(gm, halves, ws, parent_path):
    fig, ax = plt.subplots(figsize=(16,8), **{k:v for k, v in PLOT_KWARGS.items() if k not in ['figsize']})

    newcolors = np.vstack((grc_style.reds(size=128), grc_style.blues(size=128, with_alpha=True)))
    newcmp = ListedColormap(newcolors, name='GRCMap')

    im = ax.imshow(gm, cmap=newcmp, aspect='auto')

    xticks = ax.get_xticks().astype(int)
    yticks = np.arange(gm.shape[0])
    ax.set_xticks(xticks[1:-1], labels=gm.T.index.strftime('%Y-%b')[xticks[1:-1]])
    ax.set_yticks(yticks, minor=False, labels=gm.index)
    ax.set_yticks(yticks + .5, minor=True)

    halve_lines = ax.vlines(halves, yticks[0], yticks[-1], colors=grc_style.green.hash, ls='--', lw=2, label='Halving')

    ax.tick_params(axis='y', which='both', width=0)
    ax.grid(axis='y', which="minor", color="w", linestyle='-', linewidth=2)

    cbar_kw = {}
    cbarlabel = ''

    cbar = ax.figure.colorbar(im, ax=ax, fraction=.015, **cbar_kw)
    cbar_tix = cbar.ax.get_yticks()
    cbar.ax.set_yticks(cbar_tix[1:-1], labels=[f'{y:.0%}' for y in cbar_tix[1:-1]])  # vertically oriented colorbar
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.spines[:].set_visible(False)
    ax.legend(loc='best', framealpha=0.5)
    ax.set_title('Mine Gross Margin')

    render(ws, fig, parent_path, 'mine_gm', ws.name + ' Mine Level Gross Margin')
