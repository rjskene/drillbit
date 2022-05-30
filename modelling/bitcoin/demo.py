import datetime as dt
import numpy as np
import xlwings as xw

def random_time(ws, cells, max_, target):
    start = dt.datetime.now()
    i = 0
    while True:
        hash_ = np.random.randint(0, max_)
        i += 1
        ws.range(cells[0]).value = i
        ws.range(cells[1]).value = hash_
        if hash_ <= target:
            end = dt.datetime.now()
            break
    
    return (end - start).total_seconds()

@xw.sub
def mining_demo():
    wb = xw.Book.caller()
    ws = wb.sheets['Demo']
    max_ = ws.range('B2').value
    target = ws.range('B3').value
    ws.range('B6').value = random_time(ws, ['B4', 'B5'], max_, target)

@xw.sub
def mining_demo_avg():
    wb = xw.Book.caller()
    ws = wb.sheets['Demo']
    max_ = ws.range('B2').value
    target = ws.range('B3').value
    iters = ws.range('B8').value
    times = []
    for i in range(int(iters)):
        times.append(random_time(ws, ['B4', 'B5'], max_, target))
        ws.range('B9').value = i + 1
    ws.range('B10').value = np.mean(times)
