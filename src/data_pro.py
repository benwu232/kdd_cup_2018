import pandas as pd
import numpy as np
import datetime as dt
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


def complete_time(data_dict):
    for st in data_dict.keys():
        data_dict[st].UtcTime = pd.to_datetime(data_dict[st].UtcTime)
        start = data_dict[st].UtcTime.iloc[0].to_pydatetime()
        end = data_dict[st].UtcTime.iloc[-1].to_pydatetime()
        diff = end - start
        t = start
        df_cnt = 0
        while t <= end:
            if t == data_dict[st].UtcTime.iloc[df_cnt].to_pydatetime():
                t += dt.timedelta(hours=1)
                df_cnt += 1
                continue
            print(t)
            t += dt.timedelta(hours=1)
    pass

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

def build_df(bj_aq):
    data_dict = {}
    for key in bj_stations:
        data_dict[key] = []
    bj_aq.UtcTime = pd.to_datetime(bj_aq.UtcTime)
    #bj_aq.UtcTime.apply(to_pydt)
    start = dt.datetime(year=2017, month=1, day=1, hour=0)
    end = pd.to_datetime(bj_aq.UtcTime.iloc[-1]).to_pydatetime()
    for st in data_dict:
        sub_df = bj_aq[bj_aq.StationId==st]
        print('Before insert missing: {}, {}'.format(st, len(sub_df)))
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
        print('After insert missing: {}, {}'.format(st, len(data_dict[st])))

    return data_dict

def df2dict(df):
    data_dict = {}
    for st in df.StationId.unique():
        data_dict[st] = df[df.StationId == st]
    return data_dict

if __name__ == '__main__':
    bj_aq = read_bj_aq()
    check_stations(bj_aq)
    build_df(bj_aq)
    #data_dict = df2dict(bj_aq)
    #complete_time(data_dict)
    #print(bj_aq.StationId.unique())
    pass