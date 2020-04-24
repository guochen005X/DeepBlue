import numpy as np
db = np.zeros(shape=[8,3], dtype=np.float32)

for i in range(8):
    x_sign = i & 1
    y_sign = i & 2
    z_sign = i & 4
    if x_sign > 0:
        db[i,2] = 1
    if y_sign > 0:
        db[i, 1] = 1
    if z_sign > 0:
        db[i,0] = 1
    print(db[i,:])

    #print(num)




for point_indice in range(8):
    quadrant = 0
    if db[point_indice, 2] > 0:  # x轴
        quadrant = quadrant | 1
    if db[point_indice, 1] > 0:  # y轴
        quadrant = quadrant | 2
    if db[point_indice, 0] > 0:  # z轴
        quadrant = quadrant | 4
    print(quadrant)
