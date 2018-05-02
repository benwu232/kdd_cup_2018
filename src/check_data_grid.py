import pandas as pd

file_name = '../input/ld_mog_ex.csv'

df = pd.read_csv(file_name)

err_cnt = 0
for k in range(1, len(df)-1):
    last_time = df.time.iloc[k-1]
    cur_time = df.time.iloc[k]
    next_time = df.time.iloc[k+1]

    if last_time == next_time and last_time != cur_time:
        err_cnt += 1
        print(k+2, df.iloc[k].station_id, df.iloc[k].time)

print('total error: {}'.format(err_cnt))
