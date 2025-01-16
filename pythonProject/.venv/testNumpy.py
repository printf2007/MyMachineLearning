import numpy as np
import pandas as pd
# A=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
# arr1=np.array(A)
# print("A=",A)
# print("通过列表A创建的矩阵arr1\n",arr1)
# B=((1,2,3,4),(5,6,7,8),(9,10,11,12))
# arr2=np.array(B)
# print("B=",B)
# print("通过元组B创建的矩阵arr2\n",arr2)
# print("A的类型",type(arr1))
#
# arr3=np.random.random((2,3))
# print("创建随机浮点数构成的2x3阶矩阵：\n",arr3)
# arr4=np.random.randint(1,10,(2,3))
# print("1~10（不包括10）之间的随机整数构成的2x3阶矩阵：\n",arr4)
#
# arr5=arr1.reshape(4,-1)
# print("arr5 is arr1变成4x3矩阵：",arr5)
#
# print("arr5的第1行第2个元素",arr5[0,1])
# print("arr5前2行元素",arr5[1:2])
# print("arr5的第2列：",arr5[:,1])
# print("arr5的最后一列数据：",arr5[:,-1])
#
# A=[[1,2,3,4,5]]
# B=[[1],[2],[3],[4],[5]]
# C=np.array(A)
# D=np.array(B)
# print("A=\n",A)
# print("B=\n",B)
# print("C=\n",C)
# print("D=\n",D)
#
# X=[1,2,3,4,5]
# Y=np.array(X)
# print("Y=\n",Y)
# print("X的类型是：",type(X))
# print("Y的类型是：",type(Y))
# arr6=Y.reshape(1,5)
# print("arr6=\n",arr6)
#
# A1=np.array([[1,2,3,2],[4,5,6,3],[7,8,9,4],[10,11,12,5]])
# B1=np.array([[1,2,3,2],[4,5,6,3],[7,8,9,4],[10,11,12,5]])
# print("A1=\n",A1)
# upper_A1=np.triu(A1,0)
# print("upper_A1=\n",upper_A1)
# low_A1=np.tril(A1,0)
# print("low_A1=\n",low_A1)
# print("A1 and B1 is equal?",np.allclose(A1,B1))
# print("A1+B1",A1+B1)
# print("A1-B1",A1-B1)
# print("2*A1",2*A1)
#
# A2=np.array([[1,2,3],[4,5,6]])
# D2=np.array([[2,4,6],[8,10,12]])
# B2=np.array([[1,2],[3,4],[5,6]])
# C2=np.dot(A2,B2)
# E2=A2*D2
# print("C2=\n",C2)
# print("E2=\n",E2)
#
# X1=np.array([1,2,3,4,5])
# Y1=X1.reshape(1,5)
# Y2=X1.reshape(5,1)
# Z1=Y1.dot(Y2)
# print("Z1=\n",Z1)
#
# two_matrix=np.array([[1,2,3],[4,5,6]])
# vector=np.array([[7],[8],[9]])
# print("two_matrix=\n",two_matrix)
# print("vector=\n",vector)
# result=np.dot(two_matrix,vector)
# print("result=\n",result)
#
# vector2=np.array([[1,2,3]])
# A3=np.array([[1,2,3],[4,5,6],[7,8,9]])
# result2=np.dot(vector2,A3)
# print("result2=\n",result2)

# A=[[1,2,3],[4,5,6],[7,8,9]]
# A_array=np.array(A)
# A_matrix=np.mat(A)
# 创建一个 2x3 的 ndarray
# a = np.array([[1, 2, 3], [4, 5, 6]])
# print("ndarray a:")
# print(a)
#
# # 矩阵乘法（使用 @ 或者 np.dot）
# b = np.array([[7, 8], [9, 10], [11, 12]])
# c = a @ b  # 等价于 np.dot(a, b)
# print("\n矩阵乘法结果 c:")
# print(c)
#
# A=np.array([[1,2,3],[4,5,6]])
# B=np.array([[7,8],[9,10],[11,12]])
# C=np.array(A)
# D=np.array(B)
# print("矩阵相乘后的转置结果：\n",(C.dot(D)).T)
# print("矩阵转置后相乘的结果：\n",(D.T).dot(C.T))

# arr1 = np.random.randint(1, 16, (3, 3))
# arr2=np.triu(arr1,0)
# print("arr2:\n",arr2)
# arr2+=arr2.T-np.diag(np.diag(arr2))
# print("创建的方阵为:\n",arr1)
# print("生成的对称矩阵为：\n",arr2)

# 定义两个矩阵
# A = np.array([[1.0, 2.0], [3.0, 4.0]])
# B = np.array([[1.0, 2.0000001], [3.0, 4.0]])
#
# # 使用 numpy.allclose 判断两个矩阵是否几乎相等
# are_close = np.allclose(A, B,atol=1,rtol=4e-8)
#
# print(are_close)  # 输出: True
#
# A=[[1,2],[2,5]]
# C1=np.array(A)
# C2=np.asmatrix(A)
# C1_inverse=np.linalg.inv(C1)
# C2_inverse=C2.I
# print("C1_inverse=\n",C1_inverse)
# print("C2_inverse=\n",C2_inverse)
# print("C1与C1的逆相乘的结果：\n",C1.dot(C1_inverse))

# A=[[1,-4,0,2],[-1,2,-1,-1],[1,-2,3,5],[2,-6,1,3]]
# B=np.array(A)
# print("B的行列式的值：\n",np.linalg.det(B))
# A=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
# print("A的各行向量：\n")
# for i in range(A.shape[0]):
#     print(A[i,:])
# print("A的各列向量:\n")
# for i in range(A.shape[1]):
#     print(A[:,i])

# A=np.array([[1,1]])
# B=np.linalg.norm(A)
# print("向量A的长度为：\n",B)
# C=A/B
# print(A,"对应的单位向量=",C)
#
# C1=np.sum(A**2)**0.5
# print("A的长度通过定义计算为：\n",C1)
# A=np.array([[1,1,1],[2,-4,2]])
# B=np.array([[0],[0]])
# C=np.linalg.lstsq(A,B,rcond=None)
# print("A和B组成的线性方程级的解为：\n",C)

# # 定义向量 a1 和 a2
# a1 = np.array([1, 1, 1])
# a2 = np.array([2, -4, 2])
#
# # 计算 a1 和 a2 的叉积得到 a3
# a3 = np.cross(a1, a2)
#
# # 输出 a3
# print("向量 a3 为:", a3)
#
# # 验证 a3 与 a1 和 a2 是否正交
# dot_product_a1_a3 = np.dot(a1, a3)
# dot_product_a2_a3 = np.dot(a2, a3)
#
# # 输出点积结果
# print("a1 与 a3 的点积:", dot_product_a1_a3)
# print("a2 与 a3 的点积:", dot_product_a2_a3)

#
# # 定义矩阵 A
# A = np.array([[1, 1, 1],
#               [2, -4, 2]])
#
# # 进行奇异值分解
# U, s, Vt = np.linalg.svd(A)
# print("Vt的值:\n",Vt)
# # 零空间由 Vt 中对应于 s 中最小奇异值的行组成
# # 通常最后一个奇异值是最小的
# a3 = Vt[-1, :]
#
# # 输出 a3
# print("向量 a3 为:", a3)
#
# # 验证 a3 与 a1 和 a2 是否正交
# a1 = np.array([1, 1, 1])
# a2 = np.array([2, -4, 2])
#
# dot_product_a1_a3 = np.dot(a1, a3)
# dot_product_a2_a3 = np.dot(a2, a3)
#
# # 输出点积结果
# print("a1 与 a3 的点积:", dot_product_a1_a3)
# print("a2 与 a3 的点积:", dot_product_a2_a3)


# # 定义一个矩阵 A
# A = np.array([
#     [1, 1, 1],
#     [2, -4, 2]
# ])
#
# # 进行 SVD 分解
# U, s, Vt = np.linalg.svd(A)
#
# # 输出 U, s, Vt
# print("U (左奇异向量):")
# print(U)
# print("\nΣ (奇异值):")
# print(s)
# print("\nV^T (右奇异向量的转置):")
# print(Vt)
#
# # 获取 A 的秩
# rank_A = np.linalg.matrix_rank(A)
# print("\n矩阵 A 的秩:", rank_A)
#
# # 行空间的基
# row_space_basis = U[:, :rank_A]
# print("\n行空间的基:")
# print(row_space_basis)
#
# # 列空间的基
# column_space_basis = Vt[:rank_A, :].T
# print("\n列空间的基:")
# print(column_space_basis)
#
# # 零空间的基
# null_space_basis = Vt[rank_A:, :].T
# print("\n零空间的基:")
# print(null_space_basis)

# A=np.array([[0,1,0],[1/2**0.5,0,1/2**0.5],[-1/2**0.5,0,1/2**0.5]])
# print("AxA.T=\n",np.round(A.dot(A.T)))

# A=np.array([[1,2,1],[2,-1,3],[3,1,2]])
# B=np.array([[7],[7],[18]])
# C=np.linalg.solve(A,B)
# print("A和B组成的线性方程级的解为:\n",C)
#
# inverse_A=np.linalg.inv(A)
# result=inverse_A.dot(B)
# print("使用逆矩阵求A和B组成的线性方程级的解为:\n",result)
# print("inverse_A=\n",inverse_A)



#读取文件
# dataset=pd.read_excel("C:/Users/allenwang/PycharmProjects/MyMachineLearning/pythonProject/data/Folds5x2_pp.xlsx")
# #将数据转化成矩阵的形式
# data=np.array(dataset)
# #获取数据和标签
# X_data=data[:,:-1]
# Y_data=data[:,-1]
# print("数据集的样本数为%d,列数为%d"%(data.shape[0],data.shape[1]))
# #使用Numpy获取数据集的第0个样本
# data_0=X_data[0]
# print("数据集的第0个样本为：",data_0)
# print("X_data[1,0]为：",X_data[1,0])
# print("输出所有列：",dataset.columns)

# 定义矩阵 A
# A = np.array([[1, 1, 1],
#               [2, -4, 2]])
# print(pd.DataFrame(A))
# A=np.arange(1,11,1)
# print("A=\n",A)
# print("A.reshape(3,2)=\n",A.reshape(1,-1))
# print("A.shape:",A.shape)
# print("A.type",type(A))
# print("A.reshape(2,-1):\n",A.reshape(2,-1))
#
# B=np.logspace(0,6,3,base=2)
# print("B=\n",B)

# A=np.array([[4,2],[1,5]])
# print("A=\n",A)
# eig_val,eig_vex=np.linalg.eig(A)
# print("A的特征值为",eig_val)
# print("A的特征向量为",eig_vex)
# sigma=np.diag(eig_val**3)
# print("sigma=\n",sigma)
# C=eig_vex.dot(sigma.dot(np.linalg.inv(eig_vex)))
# print("C=\n",C)
# D=A.dot(A.dot(A))
# print("D=\n",D)
# print("C与D是否相同：",np.allclose(C,D))
# print("A的三次方为：",C)
from numpy import linalg as lg
A=[[1,5,7,6,1],[2,1,10,4,4],[3,6,7,5,2]]
A=np.array(A)
B=A.dot(A.T)
C=A.T.dot(A)
eig_val1,eig_vex1=lg.eig(B)
eig_val2,eig_vex2=lg.eig(C)
print("B\n",B)
print("C\n",C)
print("eig_vex1\n",eig_vex1)
print("eig_vex2\n",eig_vex2)

















