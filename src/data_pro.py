import pandas as pd
import numpy as np
import datetime as dt
import pickle
from lib.define import *

def read_bj_aq():
    bj_aq = pd.read_csv('../input/beijing_aq.csv')
    bj_aq.columns = ['StationId', 'UtcTime', 'PM25', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    print(len(bj_aq))

    bj_aq1 = pd.read_csv('../input/bj_aq_ex.csv')
    bj_aq1.drop(['id'], axis=1, inplace=True)
    bj_aq1.columns = ['StationId', 'UtcTime', 'PM25', 'PM10', 'NO2', 'CO', 'O3', 'SO2']
    print(len(bj_aq1))

    bj_aq = bj_aq.append(bj_aq1)
    bj_aq.drop_duplicates(subset=['StationId', 'UtcTime'], inplace=True)
    print(len(bj_aq))
    return bj_aq

#qianmen_aq = read_bj_aq[read_bj_aq['StationId'] == 'qianmen_aq']

def check_stations(df):
    station_list = df.StationId.unique()
    ref_len = len(df[df['StationId'] == station_list[0]])
    print('Reference length: {}'.format(ref_len))
    not_eq_cnt = 0
    for st in station_list:
        st_len = len(df[df['StationId'] == st])
        if st_len != ref_len:
            not_eq_cnt += 1
            print('{} length: {}'.format(st, st_len))

    print('not equal count: {}'.format(not_eq_cnt))


def dt2str(dt, format="%Y-%m-%d %H:%M:%S"):
    return dt.strftime(format)

def dt2pdts(dt):
    str = dt2str(dt)
    ts = pd.Timestamp(str)
    return ts

def build_df1(bj_aq):
    data_dict = {}
    for key in bj_stations:
        data_dict[key] = []
    bj_aq.UtcTime = pd.to_datetime(bj_aq.UtcTime)
    #bj_aq.UtcTime.apply(to_pydt)
    start = dt.datetime(year=2017, month=1, day=1, hour=0)
    end = pd.to_datetime(bj_aq.UtcTime.iloc[-1]).to_pydatetime()
    t = start
    df_cnt = 0
    while t <= end:
        for st in data_dict:
            ts = dt2pdts(t)
            row = bj_aq[(bj_aq.StationId==st) & (bj_aq.UtcTime==ts)]
            if len(row) > 0:
                row_dict = row.to_dict()
            else:
                row_dict = {'StationId': st, 'UtcTime': ts, 'PM25': np.nan, 'PM10': np.nan, 'NO2': np.nan, 'CO': np.nan, 'O3': np.nan, 'SO2': np.nan}
            data_dict[st].append(row_dict)
        df_cnt += 1
        if df_cnt % 100 == 0:
            print(df_cnt)
        t += dt.timedelta(hours=1)

    for st in data_dict:
        data_dict[st] = pd.DataFrame(data_dict)
        print(st, len(data_dict[st]))
    return data_dict

def build_data_dict(bj_aq):
    data_dict = {}
    for key in bj_stations:
        data_dict[key] = []
    bj_aq.UtcTime = pd.to_datetime(bj_aq.UtcTime)
    #bj_aq.UtcTime.apply(to_pydt)
    start = dt.datetime(year=2017, month=1, day=1, hour=0)
    end = pd.to_datetime(bj_aq.UtcTime.iloc[-1]).to_pydatetime()
    for st in data_dict:
        sub_df = bj_aq[bj_aq.StationId==st]
        print('Before insert missing rows: {}, {}'.format(st, len(sub_df)))
        df_cnt = 0
        t = start

        while t <= end:
            ts = dt2pdts(t)
            #row = sub_df[sub_df.UtcTime==ts]
            sub_row = sub_df.iloc[df_cnt]
            if ts < sub_row.UtcTime:
                row_dict = {'StationId': st, 'UtcTime': ts, 'PM25': np.nan, 'PM10': np.nan, 'NO2': np.nan, 'CO': np.nan, 'O3': np.nan, 'SO2': np.nan}
                t += dt.timedelta(hours=1)
            elif ts == sub_row.UtcTime:
                row_dict = sub_row.to_dict()
                t += dt.timedelta(hours=1)
                df_cnt += 1
                #if df_cnt % 1000 == 0:
                #    print(df_cnt)
            else:
                print('should not run here!')
                exit()
            data_dict[st].append(row_dict)
        data_dict[st] = pd.DataFrame(data_dict[st])
        print('After insert missing rows: {}, {}'.format(st, len(data_dict[st])))

    return data_dict

def build_bj_st():
    with open('../input/bj_st_dict.pkl', 'rb') as fp:
        data_dict = pickle.load(fp)

    data_nan_dict = {}
    for st in data_dict:
        data_dict[st] = data_dict[st][['PM25', 'PM10', 'O3', 'CO', 'NO2', 'SO2']]
        data_nan_dict[st] = pd.isnull(data_dict[st])#.astype('uint8')
        #data_nan_dict.columns = ['PM25_NAN', 'PM10_NAN', 'O3_NAN', 'CO_NAN', 'NO2_NAN', 'SO2_NAN']
        #data_dict[st] = pd.concat([data_dict[st], data_nan_dict], axis=1)
    return data_dict, data_nan_dict


def batch_gen(indices, build_batch, bb_pars={},
              batch_size=128, shuffle=False, forever=True, drop_last=True):
    data_len = len(indices)
    #indices = np.arange(data_len)

    if shuffle:
        np.random.shuffle(indices)

    while True:
        for k in range(0, data_len-batch_size, batch_size):
            excerpt = indices[k:k + batch_size]
            batch_data = build_batch(excerpt, pars=bb_pars)
            yield batch_data

        if not forever:
            break

        if shuffle:
            np.random.shuffle(indices)

    if not drop_last:
        k += batch_size
        if k < data_len:
            excerpt = indices[k:]
            batch_data = build_batch(excerpt, pars=bb_pars)
            yield batch_data

#city: 0 - Beijing; 1 - London
#type: 0 - station, 1 - grid
class DataBuilder(object):
    def __init__(self):
        #self.data = [[] for _ in (0, 1)]
        bj_st, bj_st_nan = build_bj_st()
        self.data = bj_st['aotizhongxin_aq'].values
        self.data_nan = bj_st_nan['aotizhongxin_aq'].values
        self.past_len = 128
        self.future_len = 48

    def build_batch(self, idxes, pars={}):
        with_targets = False
        if 'with_targets' in pars:
            with_targets = pars['with_targets']

        #print('SeqLen = {}'.format(self.past_len))
        data_batch = []
        data_batch_nan = []
        target_batch = []
        target_batch_nan = []
        for sti in idxes:
            data_batch.append(self.data[sti+1-self.past_len:sti+1])
            data_batch_nan.append(self.data_nan[sti+1-self.past_len:sti+1])

            if with_targets:
                target_batch.append(self.data[sti+1:sti+1+self.future_len, :3])
                target_batch_nan.append(self.data_nan[sti+1:sti+1+self.future_len, :3])

        #print(len(data_batch), type(data_batch))
        data_batch = np.asarray(data_batch, dtype=np.float32)
        data_batch = np.nan_to_num(data_batch)
        data_batch_nan = np.asarray(data_batch_nan, dtype=np.float32)
        if with_targets:
            target_batch = np.asarray(target_batch, dtype=np.float32)
            target_batch = np.nan_to_num(target_batch)
            target_batch_nan = np.asarray(target_batch_nan, dtype=np.float32)
            return data_batch, data_batch_nan, target_batch, target_batch_nan
        return data_batch, data_batch_nan




if __name__ == '__main__':
    '''
    bj_aq = read_bj_aq()
    check_stations(bj_aq)
    build_data_dict(bj_aq)
    '''
    #build_bj_st()
    data_builder = DataBuilder()
    data_builder.build_batch([200,300,400], pars={'with_targets': True})


    #complete_time(data_dict)
    #print(bj_aq.StationId.unique())
    pass