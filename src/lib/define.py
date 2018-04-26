import numpy as np
from geopy.distance import geodesic

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


bj_latitude0 = 39.0
bj_longitude0 = 115.0
bj_origin = (bj_latitude0, bj_longitude0)
ld_latitude0 = 50.5
ld_longitude0 = -2.0
ld_origin = (ld_latitude0, ld_longitude0)

def cal_pos(point, origin):
    x = geodesic(origin, (origin[0], point[1])).kilometers
    y = geodesic((point[0], origin[1]), origin).kilometers
    return x, y

#print(cal_pos((39.1, 115.1), bj_origin))
#print(cal_pos((50.6, -1.9), ld_origin))



