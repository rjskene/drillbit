import http
from pywintypes import com_error
import pandas as pd
import matplotlib.pyplot as plt

from btc.objects import HashRate, Energy
from ..style import grc_style

PLOT_KWARGS = dict(figsize=(8,5), num=1, clear=True)

def render(ws, fig, parent_path, filename, img_name):
    img_path = parent_path / 'img' / f'{filename}.png'
    plt.savefig(img_path)

    try:
        ws.pictures.add(img_path, name=ws.name + f' {img_name}', update=True, left=ws.range('E7').left)
    except com_error as e:
        print (f'Error saving {img_name}')
        print (repr(e))

    plt.close(fig)

def chart_block_sched(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)
    state.block_sched.reward.resample('D').last().plot(ax=ax)

    rewards = state.block_sched.reset_index().set_index('reward').drop(['block_id', 'retarget'], axis=1).groupby('reward').median()
    if rewards.shape != (1,1):
        rewards = rewards.sort_index(ascending=False).squeeze()
    else:
        rewards = rewards.Period

    for i, (reward, period) in enumerate(rewards.iteritems()):
        epoch = state.BTC.CURRENT_EPOCH + i
        text = f'Epoch {epoch} \n Reward {reward:.5f}'.rstrip('0')
        ax.text(period, reward + 0.1, text , ha='center', color=grc_style.green.hash)

    ax.yaxis.set_major_formatter(grc_style.btc_fmt)
    ax.set_xlabel('')
    ax.set_title('Bitcoin Reward')

    render(ws, fig, state.parent_path, 'block_sched', 'Block Schedule')

def chart_btc_forecast(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)
    state.btc_price.resample('D').mean().plot(ax=ax)

    ax.yaxis.set_major_formatter(grc_style.dol_fmt)
    ax.set_xlabel('')
    ax.set_title('Bitcoin Price Forecast')

    render(ws, fig, state.parent_path, 'btc_forecast', 'BTC Forecast')

def chart_btc_price(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)

    try:
        hist_price = state.BTC.historical_price().resample('D').mean()
        hist_price.index = pd.PeriodIndex(hist_price.index)
    except http.client.RemoteDisconnected as e:
        hist_price = pd.Series([], dtype=float)
    except Exception as e:
        raise e

    fore_price = state.btc_price.resample('D').mean()
    btc_cat = pd.concat((hist_price, fore_price))
    btc_cat.dropna().plot(ax=ax)

    ax.axvspan(fore_price.index[0].strftime('%Y-%m-%d'), fore_price.index[-1].strftime('%Y-%m-%d'), fc=grc_style.vvvlgreen.hash)

    ax.yaxis.set_major_formatter(grc_style.dol_fmt)
    ax.set_xlabel('')
    ax.set_title('BTC: Historical and Forecast')

    render(ws, fig, state.parent_path, 'btc_price', 'BTC Price')

def chart_fee_forecast(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)
    state.traxn_fees.resample('D').sum().iloc[:-1].plot(ax=ax)

    ax.yaxis.set_major_formatter(grc_style.btc_fmt)
    ax.set_xlabel('')
    ax.set_title('Transaction Fee Forecast')

    render(ws, fig, state.parent_path, 'fee_forecast', 'Fee Forecast')

def chart_fee_price(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)

    try:
        hist_fees = state.BTC.historical_transaction_fees().resample('D').mean()
        hist_fees.index = pd.PeriodIndex(hist_fees.index)
    except http.client.RemoteDisconnected as e:
        hist_fees = pd.Series([], dtype=float)
    except Exception as e:
        raise e

    fore_fees = state.traxn_fees.resample('D').sum().iloc[:-1]
    btc_cat = pd.concat((hist_fees, fore_fees))
    btc_cat.dropna().plot(ax=ax)

    ax.axvspan(fore_fees.index[0].strftime('%Y-%m-%d'), fore_fees.index[-1].strftime('%Y-%m-%d'), fc=grc_style.vvvlgreen.hash)

    ax.yaxis.set_major_formatter(grc_style.btc_fmt)
    ax.set_xlabel('')
    ax.set_title('Fees: Historical and Forecast')

    render(ws, fig, state.parent_path, 'fees', ws.name + ' Historical Fee')

def chart_mining_revenue(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)

    cols = ['Market Rewards', 'Market Fees', 'Market Revenue']
    state.env.loc[cols].T.resample('D').sum().iloc[:-1].plot(ax=ax)

    ax.yaxis.set_major_formatter(grc_style.dol_fmt)
    ax.set_xlabel('')
    ax.set_title('Total Mining Revene per Day')

    render(ws, fig, state.parent_path, 'total_mining_revenue', ws.name + ' Total Mining Revenue')

def chart_mining_revenue_comp(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)
    try:
        hist_price = state.BTC.historical_revenue().resample('D').mean()
        hist_price.index = pd.PeriodIndex(hist_price.index)
    except http.client.RemoteDisconnected as e:
        hist_price = pd.Series([])
    except Exception as e:
        raise e

    fore_price = state.env.mkt_rev.resample('D').sum()
    btc_cat = pd.concat((hist_price, fore_price))
    btc_cat.dropna().iloc[:-1].plot()

    ax.axvspan(fore_price.index[0].strftime('%Y-%m-%d'), fore_price.index[-1].strftime('%Y-%m-%d'), fc=grc_style.vvvlgreen.hash)    

    ax.yaxis.set_major_formatter(grc_style.dol_fmt)
    ax.set_xlabel('')
    ax.set_title('Revenue: Historical and Forecast')

    render(ws, fig, state.parent_path, 'total_mining_revenue_hist_comp', ws.name + ' Total Mining Revenue Comparison')

def chart_difficulty(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)

    state.env.difficulty.resample('D').mean().plot(ax=ax)

    ax.yaxis.set_major_formatter(grc_style.diff_fmt)    
    ax.set_xlabel('')
    ax.set_title('Difficulty Forecast')

    render(ws, fig, state.parent_path, 'difficulty_forecast', ws.name + ' Difficulty Forecast')

def chart_difficulty_comp(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)

    try:
        hist_diff = state.BTC.historical_difficulty().resample('D').mean()
        hist_diff.index = pd.PeriodIndex(hist_diff.index)
    except http.client.RemoteDisconnected as e:
        hist_diff = pd.Series([], dtype=float)
    except Exception as e:
        raise e

    fore_diff = state.env.difficulty.resample('D').mean()
    btc_cat = pd.concat((hist_diff, fore_diff))
    btc_cat.dropna().plot()

    ax.axvspan(fore_diff.index[0].strftime('%Y-%m-%d'), fore_diff.index[-1].strftime('%Y-%m-%d'), fc=grc_style.vvvlgreen.hash)    

    ax.yaxis.set_major_formatter(grc_style.diff_fmt)
    ax.set_xlabel('')
    ax.set_title('Difficulty: Historical and Forecast')

    render(ws, fig, state.parent_path, 'difficulty_hist_comp', ws.name + ' Difficulty Forecast Comparison')

def chart_hash_rate(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)

    state.env.net_hr.resample('D').mean().plot(ax=ax)

    yticks = ax.get_yticks()
    ax.set_yticks([ytick for ytick in yticks])

    tick1 = HashRate(yticks[0])
    ax.set_yticklabels([None] + [HashRate(ytick).__repr__().split()[0].replace('.00', '') for ytick in yticks[1:]])

    ax.set_ylabel(tick1.units, rotation=0, labelpad=20, fontsize=12)
    ax.set_xlabel('')
    ax.set_title('Network Hash Rate')

    render(ws, fig, state.parent_path, 'network_hash_rate_forecast', ws.name + ' Network Hash Rate Forecast')

def chart_hash_rate_comp(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)

    try:
        hist_hr = (state.BTC.historical_hash_rate()*1e12).resample('D').mean()
        hist_hr.index = pd.PeriodIndex(hist_hr.index)
    except http.client.RemoteDisconnected as e:
        hist_hr = pd.Series([], dtype=float)
    except Exception as e:
        raise e

    fore_hr = state.env.net_hr.resample('D').mean()
    btc_cat = pd.concat((hist_hr, fore_hr))
    btc_cat.dropna().plot()

    ax.axvspan(fore_hr.index[0].strftime('%Y-%m-%d'), fore_hr.index[-1].strftime('%Y-%m-%d'), fc=grc_style.vvvlgreen.hash)    

    yticks = ax.get_yticks()
    ax.set_yticks([ytick for ytick in yticks[1:]])

    tick1 = HashRate(yticks[-1])
    ax.set_yticklabels([None] + [HashRate(ytick).__repr__().split()[0].replace('.00', '') for ytick in yticks[2:]])

    ax.set_ylabel(tick1.units, rotation=0, labelpad=20, fontsize=12)
    ax.set_xlabel('')
    ax.set_title('Hash Rate: Historical and Forecast')

    render(ws, fig, state.parent_path, 'network_hash_rate_hist', ws.name + '  Network Hash Rate Historical Comp')

def chart_miner_comps(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)

    for project, mine in zip(state.projstats.projects, state.projstats.projects.mines):
        project.n_miners.iloc[mine.implement.start_in_blocks():].resample('D').last().iloc[:-1].plot(ax=ax, label=mine.name)

    ax.set_xlabel('')
    ax.legend()
    ax.set_title('Number of Miners')

    render(ws, fig, state.parent_path, 'project_comps_miners', 'Nubmer of Miners Project Comparison')

def chart_hr_project_comps(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)

    for lineitem, mine in zip(state.projstats.projects.by_lineitem('hr'), state.projstats.projects.mines):
        lineitem.iloc[mine.implement.start_in_blocks():].resample('D').mean().iloc[:-1].plot(ax=ax, label=mine.name)

    yticks = ax.get_yticks()
    ax.set_yticks([ytick for ytick in yticks[1:]])

    tick1 = HashRate(yticks[-1])
    ax.set_yticklabels([None] + [getattr(HashRate(ytick), tick1.units.split()[0]).__repr__().split()[0].replace('.00', '') for ytick in yticks[2:]])
    ax.set_ylabel(tick1.units, rotation=0, labelpad=20, fontsize=12)

    ax.set_xlabel('')
    ax.legend()
    ax.set_title('Hash Rate')

    render(ws, fig, state.parent_path, 'project_comps_hr', 'Hash Rate Project Comparison')

def chart_energy_project_comps(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)

    for lineitem, mine in zip(state.projstats.projects.by_lineitem('energy'), state.projstats.projects.mines):
        lineitem.iloc[mine.implement.start_in_blocks():].resample('D').sum().iloc[:-1].plot(ax=ax, label=mine.name)

    yticks = ax.get_yticks()
    ax.set_yticks([ytick for ytick in yticks[1:]])

    tick1 = Energy(yticks[-1])
    ax.set_yticklabels([None] + [getattr(Energy(ytick), tick1.units.split()[0]).__repr__().split()[0].replace('.00', '') for ytick in yticks[2:]])
    ax.set_ylabel(tick1.units, rotation=0, labelpad=20, fontsize=12)

    ax.set_xlabel('')
    ax.legend()
    ax.set_title('Daily Energy Consumption')

    render(ws, fig, state.parent_path, 'project_comps_energy', 'Energy Project Comparison')

def chart_mined_energy_project_comps(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)

    for lineitem, mine in zip(state.projstats.projects.by_lineitem('miner_energy'), state.projstats.projects.mines):
        lineitem.iloc[mine.implement.start_in_blocks():].resample('D').sum().iloc[:-1].plot(ax=ax, label=mine.name)

    yticks = ax.get_yticks()
    ax.set_yticks([ytick for ytick in yticks[1:]])

    tick1 = Energy(yticks[-1])
    ax.set_yticklabels([None] + [getattr(Energy(ytick), tick1.units.split()[0]).__repr__().split()[0].replace('.00', '') for ytick in yticks[2:]])
    ax.set_ylabel(tick1.units, rotation=0, labelpad=20, fontsize=12)

    ax.set_xlabel('')
    ax.legend()
    ax.set_title('Daily Energy Consumption - Miners')

    render(ws, fig, state.parent_path, 'project_comps_energy_miners', 'Miner Energy Project Comparison')

def chart_win_per_project_comps(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)

    for lineitem, mine in zip(state.projstats.projects.by_lineitem('win_per'), state.projstats.projects.mines):
        lineitem.iloc[mine.implement.start_in_blocks():].resample('D').mean().iloc[:-1].plot(ax=ax, label=mine.name)

    ax.yaxis.set_major_formatter(grc_style.per_fmt)
    ax.set_xlabel('')
    ax.legend()
    ax.set_title('Daily Average Win Percentage')

    render(ws, fig, state.parent_path, 'project_comps_win_per', 'Win Percentage Project Comparison')

def chart_btc_mined_project_comps(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)

    for lineitem, mine in zip(state.projstats.projects.by_lineitem('btc_mined'), state.projstats.projects.mines):
        lineitem.iloc[mine.implement.start_in_blocks():].resample('D').sum().iloc[:-1].plot(ax=ax, label=mine.name)

    ax.yaxis.set_major_formatter(grc_style.btc_fmt)
    ax.set_xlabel('')
    ax.legend()
    ax.set_title('Daily BTC Mined')

    render(ws, fig, state.parent_path, 'project_comps_btc_mined', 'BTC Mined Project Comparison')

def chart_gm_project_comps(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)

    for lineitem, mine in zip(state.projstats.projects.by_lineitem('gm'), state.projstats.projects.mines):
        lineitem.iloc[mine.implement.start_in_blocks():].resample('D').mean().iloc[:-1].plot(ax=ax, label=mine.name)

    ax.yaxis.set_major_formatter(grc_style.per_fmt)
    ax.set_xlabel('')
    ax.legend()
    ax.set_title('Daily Average Gross Margin')

    render(ws, fig, state.parent_path, 'project_comps_gm', 'Gross Margin Project Comparison')

def chart_btc_value_held_project_comps(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)
    
    for lineitem, mine in zip(state.projstats.projects.by_lineitem('btc_value_held'), state.projstats.projects.mines):
        lineitem.iloc[mine.implement.start_in_blocks():].resample('D').last().iloc[:-1].plot(ax=ax, label=mine.name)

    ax.yaxis.set_major_formatter(grc_style.mill_fmt)
    ax.set_xlabel('')
    ax.legend()
    ax.set_title('Value of BTC Held')

    render(ws, fig, state.parent_path, 'project_comps_btc_held', 'BTC Held Project Comparison')

def chart_roi_project_comps(state, ws):
    fig, ax = plt.subplots(**PLOT_KWARGS)
    
    for lineitem, mine in zip(state.projstats.projects.rois.by_lineitem('roi_held'), state.projstats.projects.mines):
        start = mine.implement.start if mine.implement.start == 0 else mine.implement.start + 1
        lineitem.iloc[start:].plot(ax=ax, label=mine.name)

    ax.yaxis.set_major_formatter(grc_style.per_fmt)
    ax.set_xlabel('')
    ax.legend()
    ax.set_title('ROI, if held')

    render(ws, fig, state.parent_path, 'project_comps_roi_held', 'ROI Held Project Comparison')
