import pandas as pd
import numba
import math
import numpy as np

@numba.jit
def smape_np(y_true, y_pred):
    tmp = (np.abs(y_pred - y_true) * 200 / (np.abs(y_pred) + np.abs(y_true)))
    tmp[np.isnan(tmp)] = 0.0
    return np.mean(tmp)


print('Reading ground truth ...')
ground_truth_df = pd.read_csv('ground_truth/solution_11_01.csv')
print('Sorting ground truth ...')
ground_truth_df = ground_truth_df.sort_values('Id')
y_true = ground_truth_df['Visits'].values

print('Reading predictions ...')
predction_df = pd.read_csv('sub.csv')
print('Sorting predictions ...')
predction_df = predction_df.sort_values('Id')
y_pred = predction_df['Visits'].values


#remove unavailable elements
available_cnt = len(y_true)
del_idxes = []
for k, gt in enumerate(y_true):
    if gt <= 0.0:
        available_cnt -= 1
        del_idxes.append(k)
print('Total elements {}, available elements {}'.format(len(y_true), available_cnt))

y_true = np.delete(y_true, del_idxes)
y_pred = np.delete(y_pred, del_idxes)


smape = smape_np(y_true, y_pred)
print(smape)


