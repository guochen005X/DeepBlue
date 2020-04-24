import numpy as np

db_size = 5

data = np.random.permutation(db_size).tolist()
class Node:
    def __init__(self, key,value, left=None, right = None):
        self.left = left
        self.right = right
        self.key = key
        self.value = value

    def __str__(self):
        return str(self.value)

# def insert(root, key, value = -1):
#     if root is None:
#         root = Node(key, value)
#     else:
#         if key < root.key:
#             root.left = insert(root.left, key, value)
#         elif key > root.key:
#             root.right = insert(root.right,key, value)
#         else:
#             pass
#     return root
#
root = None
#构建，插入
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

#搜索,不能处理没有的数
def search_recursive(root,key):
    if root.key == None or root.key == key:
        return root
    if key < root.key:
        return search_recursive(root.left,key)
    elif key > root.key:
        return search_recursive(root.right,key)
node = search_recursive(root,0)#
print(node.key)

#中序遍历

def inorder(root):#[left, root, right]
    if root is not None:
        inorder(root.left)
        print(root)
        inorder(root.right)


#前序遍历[root,left,right]
def preorder(root):
    if root is not None:
        print(root)
        preorder(root.left)
        preorder(root.right)


#后序遍历 [ left, right, root]
def postorder(root):
    if root is not None:
        postorder(root.left)
        postorder(root.right)
        print(root)

#无穷大 float('Inf')

print(float('Inf'))
print(12< float('Inf'))



