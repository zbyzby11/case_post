"""
数据预处理模块，将得到的数据与模型的输入统一
@author zxd
"""
import numpy as np
import time
if __name__ == '__main__':
    e = time.time()
    a = np.random.rand(10000000,100)
    b = np.random.rand(1,100)
    print(time.time()-e)
    s = time.time()
    #print()
    c = np.dot(a,b.T)
    print(c)
    print(c.shape)
    print(time.time()-s)