PCA

@[TOC](PCA)

# PCA背景及基本思想

我们是不是经常听到这样的对话，一位同学抱怨"这个模型的变量简直要爆炸了"，然后旁边一小哥低声的说"你先PCA一下"，小哥口中的PCA到底是什么呢？

PCA是principal components analysis首字母的缩写，中文名字叫主成分分析，顾名思义是需要分清主次把主要成分提取出来进行下一步的研究工作，同时评估仅抽取主要成分会提高哪些性能，会带来哪些后果，PCA的直接目的是数据降维。现实生活中，我们需要把很多方面因素组合在一起来综合探索他们对结果的影响，同时这些因素本身有有一些相关性，于是，有人就想能不能用一些综合性指标来整体刻画这些因素，同时是的综合后的指标相互独立。

# PCA数学原理

我们把PCA的基本思想用数学语言描述出来就是把高维向量空间的数据进行某种线性变换到低维的正交的向量空间数据，这样就达到用低维数据表示高维数据的目的，同时降维后的数据各指标是线性无关的。现在的问题关键是如何找这样的一个线性变换，这就要用到数学里面的奇异值分解，所以，我们先介绍一下奇异值分解定理。

## 奇异值分解
奇异值分解，英文名为Singular Value Decomposition，简称SVD，是一种矩阵分解的方法。除此之外，矩阵分解还有很多方法，例如特征分解（Eigendecomposition）、LU分解（LU decomposition）、QR分解（QR decomposition）和极分解（Polar decomposition）等，


# PCA算法流程

因为PCA主要依据是奇异值分解，所以PCA有着比较标准的流程，具体的

- 对特征变量进行标准化处理；
- 计算特征变量的协方差矩阵；
- 对协方差矩阵进行奇异值分解；
- 取定最主要的K个特征向量作为主成分；
- 利用主成分对原特征向量进行线性变换，变换后的向量即为所求

用流程图画出来就是

# PCA实例
本次以吴恩达机器学习第七次课的内容为蓝本实际操作一下PCA，看看PCA工程化过程。

## 导库读取数据及可视化

原数据只有2个特征，可以很好的进行二维平面可视化。

```python
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat(r"D:\项目\机器学习\吴恩达机器学习课件\CourseraML\ex7\data\ex7data1.mat") #读取数据
X = data["X"] #特征变量
fig = plt.figure(figsize=(6,4)) #新建画布
plt.scatter(X[:,0], X[:,1], color='w', marker='o', edgecolors='b', linewidths= 0.2) #样本散点图
plt.show()
```
从散点图可以看到数据大致可以投影到一条右上角倾斜的直线上，这条直线就是要找的低维投影空间。

![Figure_1 数据可视化](D:/项目/PCA/Figure_1.png)

## PCA过程

按照PCA算法流程对样本数据进行PCA处理，同时记录原数据的均值，标准差，奇异值分解的特征向量和特征值，降维后的特征向量，投影误差等。

```python
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
    return  mean, std, U, S, V, myX_rec, proj_error  
```
## PCA可视化

将二维数据PCA投影到一维空间，那么是可以可视化对应关系的。

```python
def visual_pca(myX, K): #可视化pca
    mean, std, U, S, V, myZ, myX_rec, proj_error  = run_pca(myX, K)
    proj_x ,proj_y = [myX_rec[0,0], myX_rec[-1,0]], [myX_rec[0,1], myX_rec[-1,1]] #投影空间的基向量
    ortho_x, ortho_y = [myX_rec[0,1], myX_rec[-1,1]], [myX_rec[0,1], myX_rec[-1,1]]
    plt.figure(figsize=(6,4)) #新建画布
    plt.scatter(myX[:,0], myX[:,1], color='w', marker='o', edgecolors='b', linewidths= 0.2, label = "origin dots") #原数据散点图
    plt.plot(proj_x ,proj_y, color = 'b', linewidth = 1, label = "proj_base_line") #投影空间的基向量
    #plt.plot(ortho_x, ortho_y)
    plt.scatter(myX_rec[:,0], myX_rec[:,1], color ='w', marker='o', edgecolor = 'r', linewidths=0.2, label = "proj dots") #投影空间散点
    for i in range(len(myX)): #对应关系
        plt.plot([myX[i,0], myX_rec[i,0]], [myX[i,1], myX_rec[i,1]], 'k--', linewidth = 0.4)
    plt.axis("equal") 
    plt.grid() 
    plt.legend()
    plt.show()
```
![Figure_1 PCA过程](D:/项目/PCA/Figure_2.png)

# PCA优缺点

(1) 流程清晰，操作简单

(2) 会丢掉部分信息

极端的情况，如果我们保留所有特征向量，进行投影的化，不会损失信息，但同时又达不到数据压缩的目的，所以要在在信息丢失有限的情况下，选择尽可能小的主成分数，假设希望损失的信息不超过1%,那么在选择主成分数的时候可以去计算特征值矩阵S

$$
\begin{pmatrix}
    s_{11} & 0 &\cdots & 0\\
   0 & s_{22} & \cdots & 0\\
     \vdots & \vdots & \cdots & \vdots\\
     0 & 0 & \cdots & s_{mm}\\
\end{pmatrix}
$$

让k从1取到m, 看下面式子什么时候能满足，找出最小的k值。

$$
1- \frac{\sum\limits_{i =1}^k s_{ii}}{\sum\limits_{i =1}^m s_{ii}} \leq 1\%
$$


# PCA应用场景

PCA主要用于数据降维处理，对数据压缩去噪，将重点突出，不明显的隐藏掉，主次分明。

# 参考文献

1，奇异值分解(SVD)原理与在降维中的应用
https://www.cnblogs.com/pinard/p/6251584.html