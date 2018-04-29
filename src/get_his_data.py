import requests
import pandas as pd
import datetime as dt

prefix = [
    'https://biendata.com/competition/airquality/bj/',
    'https://biendata.com/competition/meteorology/bj/',
    'https://biendata.com/competition/meteorology/bj_grid/',
    'https://biendata.com/competition/airquality/ld/',
    'https://biendata.com/competition/meteorology/ld_grid/'
    ]
surfix = '/2k0d1d8'

out_dir = '../input/'
ex_files = [
    '{}bj_aq_ex.csv'.format(out_dir),
    '{}bj_mo_ex.csv'.format(out_dir),
    '{}bj_mog_ex.csv'.format(out_dir),
    '{}ld_aq_ex.csv'.format(out_dir),
    '{}ld_mog_ex.csv'.format(out_dir),
]

def get_his_data(end='2018-04-20-23'):
    for k, fn in enumerate(ex_files):
        df = pd.read_csv(fn)
        last_time = pd.to_datetime(df.time.iloc[-1]).to_datetime()
        last_time += dt.timedelta(hours=1)
        last_time = last_time.strftime('%Y-%m-%d-%H')
        url = prefix[k] + last_time + '/' + end + surfix
        print('request {} ...'.format(url))
        respones = requests.get(url)
        if len(respones.text) < 30:
            continue
        ex_str = respones.text.split('\n', 1)[-1]
        print('write to file: {}'.format(fn))
        with open(fn, 'a') as f:
            f.write(ex_str)



def get_data420(start='2018-02-01-0', end='2018-04-20-23'):
    url_bj_aq = 'https://biendata.com/competition/airquality/bj/{}/{}/2k0d1d8'.format(start, end)
    url_bj_mo = 'https://biendata.com/competition/meteorology/bj/{}/{}/2k0d1d8'.format(start, end)
    url_bj_mog = 'https://biendata.com/competition/meteorology/bj_grid/{}/{}/2k0d1d8'.format(start, end)
    url_ld_aq = 'https://biendata.com/competition/airquality/ld/{}/{}/2k0d1d8'.format(start, end)
    url_ld_mog = 'https://biendata.com/competition/meteorology/ld_grid/{}/{}/2k0d1d8'.format(start, end)
    urls = [url_bj_aq, url_bj_mo, url_bj_mog, url_ld_aq, url_ld_mog]

    out_dir = '../input/'
    bj_aq_name = '{}bj_aq_{}.csv'.format(out_dir, end)
    bj_mo_name = '{}bj_mo_{}.csv'.format(out_dir, end)
    bj_mog_name = '{}bj_mog_{}.csv'.format(out_dir, end)
    ld_aq_name = '{}ld_aq_{}.csv'.format(out_dir, end)
    ld_mog_name = '{}ld_mog_{}.csv'.format(out_dir, end)
    file_names = [bj_aq_name, bj_mo_name, bj_mog_name, ld_aq_name, ld_mog_name]

    for url, fn in zip(urls, file_names):
        print('request {} ...'.format(url))
        respones= requests.get(url)
        print('write to file: {}'.format(fn))
        with open(fn, 'w') as f:
            f.write(respones.text)


if __name__ == '__main__':
    #get_data420(end='2018-04-20-23')
    get_his_data(end='2018-08-20-23')
