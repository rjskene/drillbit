from numpy.core._exceptions import _ArrayMemoryError

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
