#prepare historical grid data (the major grid file)

from get_his_data import get_his_data
from lib.data_pro import integrate_data0



#get_his_data()
integrate_data0('../input/data0.pkl')
from lib.data_pro import period1, period2
print(period1)
print(period2)
