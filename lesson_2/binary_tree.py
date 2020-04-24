import numpy as np
class Node:
    def __init__(self, value, left=None, right = None):
        self.left = left
        self.right = right
        self.value = value

    def __str__(self):
        return str(self.value)


A, B, C, D, E, F, G, H, I = [ Node(node_value) for node_value in 'ABCDEFGHI']

A.letf, A.right = B, C
B.right = D
C.left, C.right = E, F
E.left = G
F.left, F.right = H, I

query = np.array([1,1,1],np.float32)
leaf_points = np.random.randn(5,3)
print(leaf_points)
diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
print(diff)










# class Binary_tree:
#     def build_tree(self,Node):
#         if Node.root:
#             return True
#
#         if self.left != None:
#             bui


