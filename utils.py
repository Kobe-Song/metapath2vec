import numpy as np

class AliasTable:
    def __init__(self, probs):
        '''
        该函数根据已有的概率分布，产生符合该分布的采样，该函数功能为生成采样器
        Compute utility lists for non-uniform sampling from discrete distributions.
        Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        for details
        '''
        K = len(probs)
        self.q = np.zeros(K)                     #初始化 概率队列
        self.J = np.zeros(K, dtype=np.int)       #初始化 其他结点队列

        smaller = []                        #存储结点面积小于1的结点
        larger = []                         #存储结点面积大于等于1的结点
        for kk, prob in enumerate(probs):   #初始化队列smaller，larger
            self.q[kk] = K*prob
            if self.q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:     #生成J, q
            small = smaller.pop()
            large = larger.pop()

            self.J[small] = large
            self.q[large] = self.q[large] + self.q[small] - 1.0
            if self.q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)


    def sampling(self):
        '''
        根据采样器生成的队列，进行采样
        Draw sample from a non-uniform discrete distribution using alias sampling.
        '''
        K = len(self.J)

        kk = int(np.floor(np.random.rand()*K))      #随机生成队列下标
        if np.random.rand() < self.q[kk]:                #随机生成概率，判断是否选择该结点
            return kk
        else:
            return self.J[kk]