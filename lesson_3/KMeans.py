# 文件功能： 实现 K-Means 算法

import numpy as np
import math
class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        self.data = data
        self.num_data = data.shape[0]
        self.Dimetion = data.shape[1]
        random_idx = np.random.random_integers(0,self.num_data,self.k_)
        self.mu = self.data[random_idx,:]



        self.labels = np.zeros([self.num_data,],dtype=np.int32)
        self.loss = float('inf')


        # 屏蔽结束

    def update_label(self,init_label=None):
        data = self.data
        mu = self.mu
        if init_label == None:
            for idx in range(self.num_data):
                dis_l = []
                for i in range(self.k_):
                    diff = data[idx] - mu[i]
                    diff = np.reshape(diff, [self.Dimetion,1])
                    dis = np.matmul(diff.T,diff)
                    # dis = math.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
                    dis_l.append(dis[0,0])
                self.labels[idx] = np.argmin(dis_l)




    def update_loss(self):
        data = self.data
        mu = self.mu
        labels = self.labels
        loss = 0
        for idx in range(self.num_data):
            diff = data[idx] - mu[labels[idx]]
            diff = np.reshape(diff, [self.Dimetion, 1])
            dis = np.matmul(diff.T, diff)
            loss += dis[0,0]
        return loss



    def update_k_center(self):
        for i in range(self.k_):
            idx = np.where(self.labels == i)[0]
            data = self.data[idx]
            self.mu[i] = np.mean(data, axis=0)






    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        for i in range(self.max_iter_):
            self.update_label()
            loss = self.update_loss()
            if loss < self.loss:
                self.loss = loss
                #print('self.loss = ',self.loss)
            else:
                print('loss = ', loss)
                break
            self.update_k_center()

        print(self.labels)

        result = self.labels
        # 屏蔽结束
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)

