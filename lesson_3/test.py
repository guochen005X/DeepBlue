from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math

mu  = [0,0]
var = [1,1]
def prb_2d_guass(data_x, data_y):
    data_num = data_x.shape[0]
    z = np.zeros([data_x.shape[0], data_y.shape[0]],dtype=np.float32)
    for i in range(data_num):
        for j in range(data_y.shape[0]):
            if i == 19 and j == 19:
                print(data_x[i], data_y[j])
            x_u1 = data_x[i] - mu[0]
            x_u2 = data_y[j] - mu[1]
            data = np.array([x_u1,x_u2],dtype=np.float32)
            data = data.reshape([2,1])
            sigma = np.diag(var)

            P = np.exp(-0.5* np.matmul(np.matmul(data.T,np.linalg.inv(sigma)),  data ) )
            #P = np.exp(-0.5* (x_u1*x_u1/var[0] + x_u2*x_u2/var[1] ))
            P = P/(2*math.pi * math.sqrt(np.linalg.det(sigma)))#math.sqrt(var[0] * var[1])
            z[i,j] = P[0,0]
    return z

fig = plt.figure()
ax = Axes3D(fig)
len = 8;
step = 0.4;


def build_layer(z_value):
 x = np.arange(-len, len, step);
 y = np.arange(-len, len, step);
 z1 = np.full(x.size, z_value/2)
 z2 = np.full(x.size, z_value/2)
 z1, z2 = np.meshgrid(z1, z2)
 z = z1 + z2;

 x, y = np.meshgrid(x, y)
 return (x, y, z);

def build_gaussian_layer(mean, standard_deviation):
 x = np.arange(-len, len, step);
 y = np.arange(-len, len, step);
 z = prb_2d_guass(x, y)
 x, y = np.meshgrid(x, y);
 # z = np.exp(-((y-mean)**2 + (x - mean)**2)/(2*(standard_deviation**2)))
 # z = z/(np.sqrt(2*np.pi)*standard_deviation);
 return (x, y, z);

# 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
x1, y1, z1 = build_layer(0.2);
ax.plot_surface(x1, y1, z1, rstride=1, cstride=1, color='green')

x5, y5, z5 = build_layer(0.15);
ax.plot_surface(x5, y5, z5, rstride=1, cstride=1, color='pink')


x3, y3, z3 = build_gaussian_layer(0, 1)
ax.plot_surface(x3, y3, z3, rstride=1, cstride=1, cmap='rainbow')
print(np.sum(z3))
plt.show()


#这是第三张图片的代码

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

x, y = np.mgrid[-1:1:20j, -1:1:20j]
z = x * np.exp(-x ** 2 - y ** 2)

ax = plt.subplot(111, projection='3d')
ax.plot_surface(x, y, z, rstride=2, cstride=1, cmap=plt.cm.coolwarm, alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()



import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib import style
style.use('fivethirtyeight')
mu_params = [[0,0]]
sd_params = [[1,1]]
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)


z = stats.norm(mu_params, sd_params).pdf((x[50],y[50]))


print(z)

from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(1)
x1, y1 = make_circles(n_samples=1000, factor=0.5, noise=0.1)
# datasets.make_circles()专门用来生成圆圈形状的二维样本.factor表示里圈和外圈的距离之比.每圈共有n_samples/2个点，、
plt.subplot(121)
plt.title('make_circles function example')
plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1)

plt.subplot(122)
x1, y1 = make_moons(n_samples=1000, noise=0.1)
plt.title('make_moons function example')
plt.scatter(x1[:, 0], x1[:, 1], marker='o', c=y1)
plt.show()
