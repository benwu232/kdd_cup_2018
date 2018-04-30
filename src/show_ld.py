
import numpy as np
import scipy
import matplotlib.pyplot as mplt

from lib.data_pro import build_bj_st
from lib.define import load_dump

raw_data = load_dump('../input/data.pkl')

#data = np.nan_to_num(data)

idx2name = {0: 'PM25_', 1:'PM10_', 2: 'O3_'}
#time = list(range(data[1].shape[1]))
for st in raw_data[1][0]:
    data = raw_data[1][0][st]
    mplt.title(st)
    time = list(range(len(data)))
    mplt.plot(time, np.log(data.PM25.values), 'r', time, np.log(data.PM10.values), 'g')
    mplt.show()

pass
