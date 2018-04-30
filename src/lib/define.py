import numpy as np
import datetime as dt
import torch
from geopy.distance import geodesic
import logging
import pickle

DBG = 0

DECODE_STEPS = 48
USE_CUDA = True
device = torch.device("cuda" if USE_CUDA else "cpu")

bj_stations = [
    'aotizhongxin_aq', 'badaling_aq', 'beibuxinqu_aq', 'daxing_aq',
    'dingling_aq', 'donggaocun_aq', 'dongsi_aq', 'dongsihuan_aq',
    'fangshan_aq', 'fengtaihuayuan_aq', 'guanyuan_aq', 'gucheng_aq',
    'huairou_aq', 'liulihe_aq', 'mentougou_aq', 'miyun_aq',
    'miyunshuiku_aq', 'nansanhuan_aq', 'nongzhanguan_aq',
    'pingchang_aq', 'pinggu_aq', 'qianmen_aq', 'shunyi_aq',
    'tiantan_aq', 'tongzhou_aq', 'wanliu_aq', 'wanshouxigong_aq',
    'xizhimenbei_aq', 'yanqin_aq', 'yizhuang_aq', 'yongdingmennei_aq',
    'yongledian_aq', 'yufa_aq', 'yungang_aq', 'zhiwuyuan_aq']

ld_stations = ['BL0', 'CD9', 'CD1', 'GN0', 'GR4', 'GN3', 'GR9', 'HV1', 'KF1', 'LW2', 'ST5', 'TH4',
               'MY7', 'BX9', 'BX1', 'CT2', 'CT3', 'CR8', 'GB0', 'HR1', 'LH0', 'KC1', 'RB7', 'TD5']

bj_latitude0 = 39.0
bj_longitude0 = 115.0
bj_origin = (bj_latitude0, bj_longitude0)
ld_latitude0 = 50.5
ld_longitude0 = -2.0
ld_origin = (ld_latitude0, ld_longitude0)
origin_list = [bj_origin, ld_origin]

def cal_pos(point, origin):
    x = geodesic(origin, (origin[0], point[1])).kilometers
    y = geodesic((point[0], origin[1]), origin).kilometers
    return x, y

#print(cal_pos((39.1, 115.1), bj_origin))
#print(cal_pos((50.6, -1.9), ld_origin))


def now2str(format="%Y-%m-%d_%H-%M-%S-%f"):
    return dt.datetime.now().strftime(format)

def save_dump(dump_data, out_file):
    with open(out_file, 'wb') as fp:
        print('Writing to %s.' % out_file)
        #pickle.dump(dump_data, fp, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dump_data, fp)

def load_dump(dump_file):
    fp = open(dump_file, 'rb')
    if not fp:
        print('Fail to open the dump file: %s' % dump_file)
        return None
    dump = pickle.load(fp)
    fp.close()
    return dump

def init_logger(name='td', to_console=True, log_file=None, level=logging.DEBUG,
                msg_fmt='[%(asctime)s]  %(message)s', time_fmt="%Y-%m-%d %H:%M:%S"):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create formatter
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    formatter = logging.Formatter(msg_fmt, time_fmt)

    if logger.handlers != [] and isinstance(logger.handlers[0], logging.StreamHandler):
        logger.handlers.pop(0)
    # create console handler and set level to debug
    f = open("/tmp/debug", "w")          # example handler
    if to_console:
        f = None

    ch = logging.StreamHandler(f)
    ch.setLevel(level)
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, mode='a', encoding=None, delay=False)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
