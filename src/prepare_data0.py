#prepare historical grid data (the major grid file)

from get_his_data import get_his_data
from lib.data_pro import integrate_data0
from check_download_data import check_download
from lib.define import *

#get_his_data()
#check_download()
integrate_data0('../input/data_aq_ex0.pkl')
