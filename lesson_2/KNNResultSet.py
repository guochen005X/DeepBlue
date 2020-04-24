import numpy as np
import copy

class Worst_index:
    def __init__(self, dis, index):#index原数据集中的索引
        self.worst_dis = dis
        self.index = index

class KNNResultSet:
    def __init__(self,capacity):
        self.worst_dis = float('Inf')
        self.count = 0
        self.capacity = capacity
        self.worst_dis_index = [Worst_index(self.worst_dis, 0) for i in range(capacity)]

    def size(self):
        return self.count

    def full(self):
        return self.count == self.capacity

    def worstDis(self):
        return self.worst_dis


    def add_point(self,dis,index):
        if dis > self.worst_dis:
            return

        if self.count < self.capacity:
            self.count += 1

        inv_start = self.count -1
        while inv_start > 0:
            if dis < self.worst_dis_index[inv_start -1].worst_dis:
                self.worst_dis_index[inv_start] = copy.deepcopy(self.worst_dis_index[inv_start -1])
                inv_start -= 1
            else:
                break

        self.worst_dis_index[inv_start].worst_dis = dis
        self.worst_dis_index[inv_start].index = index
        self.worst_dis = self.worst_dis_index[self.capacity-1].worst_dis


