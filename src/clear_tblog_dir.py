import os
import shutil

from lib.define import get_dir_size




def clear_dir(root_dir, min_size=1000):
    del_cnt = 0
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            dir_size = get_dir_size(entry.path)
            if dir_size < min_size:
                try:
                    shutil.rmtree(entry.path)
                    print('{} deleted.'.format(entry.path))
                    del_cnt += 1
                except Exception as ex:
                    print(ex)
                    #print ("Error: %s - %s." % (e.filename,e.strerror))
    print('Mission accomplished! {} directories are deleted.'.format(del_cnt))


if __name__ == '__main__':
    clear_dir('../tblog', 1000)