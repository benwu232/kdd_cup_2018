
import numpy as np
import scipy
import matplotlib.pyplot as mplt

from lib.data_pro import build_bj_st

def acf(y, title=None):
    y = np.nan_to_num(y)
    yunbiased = y-np.mean(y)
    ynorm = np.sum(yunbiased**2)
    acor = np.correlate(yunbiased, yunbiased, "same")/ynorm
    # use only second half
    half = len(acor)//2
    #acor = acor[half:half+100]
    acor = acor[half:]

    mplt.title(title)
    mplt.plot(acor)
    mplt.show()

def acf2(x, seq_len=24, title=None):
    seq = x[-seq_len:]
    seq = (seq - seq.mean()) / seq.std()
    seq_sqr = np.dot(seq, seq)
    corr_list = []
    for k in range(len(x)-seq_len, -1, -1):
        shift = x[k:k+seq_len]
        shift = (shift - shift.mean()) / shift.std()
        corr = np.dot(seq, shift) / seq_sqr
        corr_list.append(corr)
    return corr_list


data = build_bj_st()

data = np.nan_to_num(data)

idx2name = {0: 'PM25_', 1:'PM10_', 2: 'O3_'}
for st in range(data.shape[0]):
    corr_array = []
    for feature in range(3):
        #acf(data[st, :, feature], title=idx2name[feature] + str(st))
        corr_list = acf2(data[st, :, feature])
        corr_array.append(corr_list)

    mplt.title(str(st))
    time = list(range(len(corr_list)))
    mplt.plot(time, corr_array[0], 'r', time, corr_array[1], 'g', time, corr_array[2], 'b')
    mplt.show()

pass
