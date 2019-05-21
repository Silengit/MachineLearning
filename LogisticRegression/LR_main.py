# Author: Chen Xiang

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class myLogisticRegression:
    
    def __init__(self, solver='newton-cg', multi_class='ovr', max_iter = 100):
        # 我们暂时只提供了牛顿迭代法和OVR策略
        self.solver = solver
        self.multi_class = multi_class
        self.max_iter = max_iter
        
    def cmpt_P1(X_hat, beta):
        return  (1 - (1 / (1 + np.exp(X_hat.dot(beta)))))
        # P1是p1_i的列向量
    
    def classify_2(self, X, y):
        # 这是一个用来二分类的函数
        m = int(X.size / X[0].size) 
        # m是训练集的size
        one = np.ones(m)
        X_hat = np.column_stack((X, one))
        beta = np.zeros(X[0].size + 1)
        # 这里我们分别得到了x_hat的矩阵形式和初始化的beta
        for i in range(self.max_iter):
            P1 = myLogisticRegression.cmpt_P1(X_hat, beta)
            delta = -X_hat.T.dot(y - P1)
            # delta指一阶导数
            if np.inner(delta, delta) <= 1e-10:
            # print("iterating time is %d" %i)
                break
            delta2 = (P1 * (1 - P1) * X_hat.T).dot(X_hat)
            # delta2指海森矩阵，即二阶导数
            beta -= (np.linalg.inv(delta2)).dot(delta)
            # beta的更新公式
        return beta
        
    def fit(self, X, y):
        c = np.unique(y)
        # c是类别向量，其大小为n
        n_class = c.size
        m = int(X.size / X[0].size)
        self.Beta = np.zeros((n_class, X[0].size + 1))
        # 这里的大Beta是指n个分类器分别训练出来的beta构成的矩阵
        for i in range(n_class):
            z = np.zeros(m)
            # z是仅当一个类为正类时的二值标签向量
            for j in range(m):
                if y[j] == c[i]:
                    z[j] = 1
            self.Beta[i] = self.classify_2(X, z)
            
    def predict(self, X):
        m = int(X.size / X[0].size) 
        one = np.ones(m)
        X_hat = np.column_stack((X, one))
        k = np.argmax(X_hat.dot(self.Beta.T), axis=1) + 1
        # k是预测的类别，这里用下标加一来代替
        return k

def main():
    train_data = np.loadtxt('train_set.txt')
    test_data = np.loadtxt('test_set.txt')
    X1 = train_data[:, :16]
    y1 = train_data[:, 16]
    X2 = test_data[:, :16]
    y2 = test_data[:, 16]
    clf = myLogisticRegression()
    clf.fit(X1,y1)
    print("accuracy = %.4f" %(accuracy_score(y2, clf.predict(X2))))
    print("micro Precision = %.4f" %(precision_score(y2, clf.predict(X2), 
                                                     average='micro')))
    print("micro Recall = %.4f" %(recall_score(y2, clf.predict(X2), 
                                               average='micro')))
    print("micro f1 = %.4f" %(f1_score(y2, clf.predict(X2), 
                                       average='micro')))
    print("macro Precision = %.4f" %(precision_score(y2, clf.predict(X2), 
                                                     average='macro')))
    print("macro Recall = %.4f" %(recall_score(y2, clf.predict(X2), 
                                               average='macro')))
    print("macro f1 = %.4f" %(f1_score(y2, clf.predict(X2), 
                                       average='macro')))
    
if __name__ == '__main__':
    main()