from numpy.core._exceptions import _ArrayMemoryError
import dill as pickle

import finstat as fs

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
            if exc_type is _ArrayMemoryError:
                raise Exception(exc_value).with_traceback(exc_tb)
            else:
                raise exc_type(exc_value).with_traceback(exc_tb)
        
        if self.pbar is not None and self.update:
            self.pbar.update(1)

def fs_write_pickle(finstat, fname):
    nodes = {}

    for nodename in finstat._graph.nodes:
        node = finstat._graph.nodes[nodename]
        nodes[nodename] = {}
        nodes[nodename]['values'] = node['obj']().values if callable(node['obj']) else node['obj'].values
        nodes[nodename]['name'] = node['obj'].name
        nodes[nodename]['short_name'] = node['short_name']
        nodes[nodename]['position'] = node['position']
        nodes[nodename]['hide'] = node['hide']

    nodes['env_params'] = dict(
        periods=finstat._graph.graph['periods'],
        no_model=finstat._graph.graph['no_model'],
        groupby=finstat._graph.graph['by'],
        name=finstat.name,
        short_name=finstat.short_name
    )
    
    with open(fname, 'wb') as file_:
        pickle.dump(nodes, file_)

def fs_read_pickle(fname):
    with open(fname, 'rb') as file_:
        env = pickle.load(file_)

    params = env.pop('env_params')
    new_env = fs.FinancialStatement(**params)

    for nodename, node in env.items():
        values = node.pop('values')
        position = node.pop('position')
        new_env.add_account(values, **node)
        new_env.G.nodes[nodename]['position'] = position

    return new_env