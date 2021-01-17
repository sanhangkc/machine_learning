# -*- encoding: utf-8 -*-
'''
@Project :   PCA
@Desc    :   PCA算法实例
@Time    :   2021/01/17 13:39:44
@Author  :   帅帅de三叔,zengbowengood@163.com
'''

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat(r"D:\项目\机器学习\吴恩达机器学习课件\CourseraML\ex7\data\ex7data1.mat") #读取数据
X = data["X"] #特征变量
fig = plt.figure(figsize=(6,4)) #新建画布
plt.scatter(X[:,0], X[:,1], color='w', marker='o', edgecolors='b', linewidths= 0.2) #样本散点图
plt.show()

def run_pca(myX, K): #PCA过程
    mean = np.mean(myX, axis = 0) #每个特征平均值
    std = np.std(myX, axis= 0, ddof = 1) #每个特征的标准差，ddof = 1 表示在标准差计算分母是n-1
    myX_norm = (myX-mean)/std #标准化处理
    cov_mat =  np.dot(myX_norm.T, myX_norm) #协方差矩阵
    U, S, V = np.linalg.svd(cov_mat) #奇异值分解
    U_reduced = U[:,:K]  #取前K列的特征向量
    myZ = np.dot(myX_norm, U_reduced) #投影到K维空间
    myX_rec = np.dot(myZ, U_reduced.T)*std + mean #back计算特征的近似值
    proj_error = np.linalg.norm(myX-myX_rec) #计算投影误
    return  mean, std, U, S, V, myZ, myX_rec, proj_error      

def visual_pca(myX, K): #可视化pca
    mean, std, U, S, V, myZ, myX_rec, proj_error  = run_pca(myX, K)
    proj_x ,proj_y = [myX_rec[0,0], myX_rec[-1,0]], [myX_rec[0,1], myX_rec[-1,1]] #投影空间的基向量
    plt.figure(figsize=(6,4)) #新建画布
    plt.scatter(myX[:,0], myX[:,1], color='w', marker='o', edgecolors='b', linewidths= 0.2, label = "origin dots") #原数据散点图
    plt.plot(proj_x ,proj_y, color = 'b', linewidth = 1, label = "proj_base_line") #投影空间的基向量
    plt.scatter(myX_rec[:,0], myX_rec[:,1], color ='w', marker='o', edgecolor = 'r', linewidths=0.2, label = "proj dots") #投影空间散点
    for i in range(len(myX)): #对应关系
        plt.plot([myX[i,0], myX_rec[i,0]], [myX[i,1], myX_rec[i,1]], 'k--', linewidth = 0.4)
    plt.axis("equal") 
    plt.grid() 
    plt.legend()
    plt.show()

if __name__=="__main__":
    #run_pca(X, 1)
    visual_pca(X, 1)
