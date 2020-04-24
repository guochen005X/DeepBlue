from KNNResultSet import KNNResultSet
from build_BT import Node
import math
import numpy as np


def knn_search(root: Node, result_set: KNNResultSet, key):
    if root is None:
        return False

    result_set.add_point(math.fabs(root.key - key), root.value)#math.fabs(root.key - key) 距离
    if result_set.worst_dis == 0:
        return True


    if root.key >= key:#已经知道根的值大于比较的key,那么
        if knn_search(root.left, result_set, key): #如果这个为True，那么一定是满足了result_set.worst_dis == 0，但是一般情况是不可能的
            #所以一般到最后一定是root is None, 返回False.那么就判断右边。
            #但是如果math.fabs(root.key - key) < result_set.worstDis()就不需要了，直接返回False
            return True
        elif math.fabs(root.key - key) < result_set.worstDis():
            #root.key >= key 同时diff差值 < 最差距离 那么在右边子树就可能存在更小的diff
            #如果 diff差值 > 最差距离 那就没必要搜索右边了，就会在24行直接返回false
            return knn_search(root.right, result_set, key)
        return False
    else:
        if knn_search(root.right, result_set, key):
            return True
        elif math.fabs(root.key - key) < result_set.worstDis():
            return knn_search(root.left, result_set, key)
        return False




#构建二叉树
db_size = 10

data = np.random.permutation(db_size).tolist()
print('origin data : ',data)
root = None
def insert(root, key, index=-1):
    if root == None:
        return Node(key,index)
    elif key < root.key:
        root.left = insert(root.left, key, index)
    elif key > root.key:
        root.right = insert(root.right,key,index)
    return root

for i, point in enumerate(data):
    root = insert(root, point,i)

search_results = KNNResultSet(2)
knn_search(root,search_results, 3)
for result in search_results.worst_dis_index:
    print(result.worst_dis,'index = ', result.index)