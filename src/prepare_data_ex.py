from get_his_data import get_his_data
from lib.data_pro import integrate_data_ex, grid_to_aq
from lib.define import now2str, load_dump



get_his_data()
print(now2str())
integrate_data_ex('../input/data_aq_ex1.pkl')
print(now2str())
#data_ex = load_dump('../input/data1.pkl')
#grid_to_aq(data_ex)


