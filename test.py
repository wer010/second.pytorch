import sys
print(sys.path)

# ['/home/lanhai/Projects/second.pytorch', '/home/lanhai/pycharm-community-2018.2.3/helpers/pydev', '/home/lanhai/Projects/second.pytorch', '/home/lanhai/pycharm-community-2018.2.3/helpers/pydev', '/home/lanhai/.PyCharmCE2018.2/system/cythonExtensions', '/home/lanhai/anaconda3/envs/pytorch/lib/python37.zip', '/home/lanhai/anaconda3/envs/pytorch/lib/python3.7', '/home/lanhai/anaconda3/envs/pytorch/lib/python3.7/lib-dynload', '/home/lanhai/anaconda3/envs/pytorch/lib/python3.7/site-packages', '/home/lanhai/anaconda3/envs/pytorch/lib/python3.7/site-packages/IPython/extensions']
# ['/home/lanhai/Projects/second.pytorch',                                                          '/home/lanhai/Projects/second.pytorch',                                                                                                                   '/home/lanhai/anaconda3/envs/pytorch/lib/python37.zip', '/home/lanhai/anaconda3/envs/pytorch/lib/python3.7', '/home/lanhai/anaconda3/envs/pytorch/lib/python3.7/lib-dynload', '/home/lanhai/anaconda3/envs/pytorch/lib/python3.7/site-packages']


from numba import vectorize, float32
import numpy as np
import time

@vectorize([float32(float32, float32)], target='cuda')
def g(x, y):
    return x + y

def main():
    N = 32000000
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    t0 = time.process_time()

    C = g(A, B)

    t1 = time.process_time()
    delta_t = t1 - t0

    print('g executed in {0} seconds'.format(delta_t))

if __name__ == '__main__':
    main()