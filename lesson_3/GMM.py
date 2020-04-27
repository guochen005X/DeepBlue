# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')



class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def prb_2d_guass(self,input_point,mu, var):
        x_u1 = input_point[0] - mu[0]
        x_u2 = input_point[1] - mu[1]
        P = np.exp(-0.5 * (x_u1 * x_u1 / var[0] + x_u2 * x_u2 / var[1]))
        P = P / (2 * math.pi * math.sqrt(var[0] * var[1]))
        return P
    
    # 屏蔽开始
    # 更新W = Nk
    def update_Nk_gama(self):
        data_num    = self.data.shape[0]
        cluster_num = self.n_clusters
        cur_pi  = self.weight_pi
        cur_mu  = self.mu
        cur_var = self.var

        for data_idx in range(data_num):
            cur_data = self.data[data_idx]
            P_down = 0
            for cluster in range(cluster_num):
                prob_data = self.prb_2d_guass(cur_data, cur_mu[cluster],cur_var[cluster])
                P_down += cur_pi[cluster] * prob_data
            for cluster in range(cluster_num):
                P_up = cur_pi[cluster] * self.prb_2d_guass(cur_data, cur_mu[cluster],cur_var[cluster])
                self.gama[data_idx, cluster] = P_up / P_down
        self.Nk = np.sum(self.gama, axis= 0)




    # 更新pi
 
        
    # 更新Mu
    def update_mu(self):
        gama0 = self.gama[:, 0]
        gama1 = self.gama[:, 1]
        gama2 = self.gama[:, 2]
        assert self.num_data == gama0.shape[0]
        new_mu0 = np.array([0.0, 0.0], dtype= np.float32)
        new_mu1 = np.array([0.0, 0.0], dtype=np.float32)
        new_mu2 = np.array([0.0, 0.0], dtype=np.float32)
        for i in range(self.num_data):
            new_mu0 += gama0[i] * self.data[i]#这一类的高斯分布均值
            new_mu1 += gama1[i] * self.data[i]
            new_mu2 += gama2[i] * self.data[i]

        self.mu[0] = new_mu0/np.sum(gama0)
        self.mu[1] = new_mu1 / np.sum(gama0)
        self.mu[2] = new_mu2 / np.sum(gama0)




    # 更新Var


    # 屏蔽结束
    
    def fit(self, data):
        # 作业3
        # 屏蔽开始,init_mu，可以随机选三个数据点
        self.data = data
        self.num_data = self.data.shape[0]
        self.mu  = np.random.randn(self.n_clusters, data.shape[1])
        self.var = np.fabs(np.random.randn(self.n_clusters, data.shape[1]))
        weight_pi       = 1 / self.n_clusters #权重
        self.weight_pi = [ weight_pi for i in range(self.n_clusters)]
        self.gama  = np.zeros([data.shape[0], self.n_clusters],np.float32)# N * 3
        self.Nk = np.zeros([0,0,0],dtype=np.float32)



        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        print('x')
        self.update_Nk_gama()
        self.update_mu()

        # 屏蔽结束

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化

    

