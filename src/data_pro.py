import pandas as pd
import numpy as np
import datetime as dt

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



def df2dict(df):
    data_dict = {}
    for st in df.StationId.unique():
        data_dict[st] = df[df.StationId == st]
    return data_dict

if __name__ == '__main__':
    bj_aq = read_bj_aq()
    check_stations(bj_aq)
    #data_dict = df2dict(bj_aq)
    #complete_time(data_dict)
    #print(bj_aq.StationId.unique())
    pass