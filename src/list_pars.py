import os

from lib.define import  *

sb_len = 31

scoreboard = load_dump(os.path.join(clf_dir, 'scoreboard.pkl'))

#predict multiple times
for k, item in enumerate(scoreboard[:sb_len]):
    print('Generating single submission file')
    prefix = item[-1]
    par_file = prefix + '.par'
    print(par_file)
    pars = load_dump(par_file)
    print(pars)
    pass

