import os
import pandas as pd
from lib.define import load_dump
from predict import predict
import glob

submit_dir = '../submit_attn_pos'
fusion_list = glob.glob('{}/submit*.csv'.format(submit_dir))
redundant = '{}/submit.csv'.format(submit_dir)
if redundant in fusion_list:
    fusion_list.remove(redundant)

def fusion(fusion_file_list, submission_csv):
    print('Fusioning ...')
    df_sum = pd.read_csv('../input/sample_submission.csv')
    df_sum = df_sum.sort_values('test_id')
    for k, file_name in enumerate(fusion_file_list):
        fusion_file = os.path.join(submit_dir, file_name)
        df_one = (pd.read_csv(fusion_file))
        print('Sorting ...', fusion_file)
        df_one = df_one.sort_values('test_id')

        df_sum['PM2.5'] = df_sum['PM2.5'].values + df_one['PM2.5'].values
        df_sum['PM10'] = df_sum['PM10'].values + df_one['PM10'].values
        df_sum['O3'] = df_sum['O3'].values + df_one['O3'].values
        '''
        continue

        for k in range(len(df_sum)):
            if k % 1000 == 0:
                print(k)
            if df_sum.iloc[k]['Id'] != df_one.iloc[k]['Id']:
                print('Wrong at {}, sum_id: {}, one_id: {}'.format(k, df_sum.iloc[k]['Id'], df_one.iloc[k]['Id']))
            df_sum_copy.set_value(k, 'Visits', df_sum.iloc[k]['Visits'] + df_one.iloc[k]['Visits'])
            df_sum_copy.set_value(k, 'Id', df_sum.iloc[k]['Id'])
        print(df_sum_copy.iloc[:10])
        exit()
        df_sum = df_sum_copy.sort_values('Id')
        df_sum_copy = df_sum.copy()
        #df_sum['Visits'] += df_one['Visits']
        '''

    df_sum['PM2.5'] /= len(fusion_file_list)
    df_sum['PM10'] /= len(fusion_file_list)
    df_sum['O3'] /= len(fusion_file_list)

    print('Writing to {}'.format(submission_csv))
    #df_sum.to_csv(submission_csv, index=False, float_format='%.3f')
    df_sum.to_csv(submission_csv, index=False)
    print('Bingo!')



scoreboard = load_dump('../clf_attn_pos/scoreboard.pkl')

#predict multiple times
for k, item in enumerate(scoreboard):
    print('Generating single submission file')
    prefix = item[-1]
    enc_file = prefix + '_enc.pth'
    dec_file = prefix + '_dec.pth'
    out_file = '{}/submit{}.csv'.format(submit_dir, str(k+500))
    print(prefix, out_file)
    predict(enc_file, dec_file, out_file)

fusion(fusion_list, submission_csv='{}/submission.csv'.format(submit_dir))
