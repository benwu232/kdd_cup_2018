from get_his_data import get_his_data
from lib.data_pro import integrate_data



#get_his_data()
integrate_data('../input/data.pkl')
from lib.data_pro import period1, period2
print(period1)
print(period2)
