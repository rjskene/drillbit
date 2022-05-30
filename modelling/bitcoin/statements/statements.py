import finstat as fs

def init_enviro(block_sched, fees, btc):
    stat = fs.FinancialStatement(name='BTC World', periods=block_sched.index)

    stat.add_account(block_sched.index.astype('int'), name='Block ID', short_name='block_id')
    stat.add_account(block_sched.reward, name='Block Reward', short_name='reward')
    stat.add_account(fees, name='Block Fees', short_name='fees')
    stat.add_account(btc, name='BTC Price', short_name='btc_price')

    stat.add_account(fs.arr.multiply(stat.reward, stat.btc_price), name='Market Rewards', short_name='mkt_rewards')
    stat.add_account(fs.arr.multiply(stat.fees, stat.btc_price), name='Market Fees', short_name='mkt_fees')
    stat.add_account(fs.arr.add(stat.mkt_rewards, stat.mkt_fees), name='Market Revenue', short_name='mkt_rev')

    return stat
