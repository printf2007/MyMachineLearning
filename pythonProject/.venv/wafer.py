import numpy as np
size=10
x_center,y_center=size/2,size/2
radius=4

#创建坐标风格
x=np.arange(size).reshape(1,size)
print("x",x)
y=np.arange(size).reshape(size,1)
print("y",y)
X,Y=np.meshgrid(x,y)
distance_from_center=np.sqrt((X-x_center)**2+(Y-y_center)**2)
print("distance_from_center",distance_from_center)
circle_matrix=distance_from_center<=radius
result_matrix=circle_matrix.astype(int)
print("result_matrix:",result_matrix)