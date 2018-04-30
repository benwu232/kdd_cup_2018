
import numpy as np
import scipy
import matplotlib.pyplot as mplt

from lib.data_pro import build_bj_st



data = build_bj_st()

data = np.nan_to_num(data)

idx2name = {0: 'PM25_', 1:'PM10_', 2: 'O3_'}
time = list(range(data.shape[1]))
for st in range(data.shape[0]):
    mplt.title(str(st))
    mplt.plot(time, np.log1p(data[st, :, 0]), 'r', time, np.log1p(data[st, :, 1]), 'g', time, np.log1p(data[st, :, 2]), 'b')
    mplt.show()

pass
