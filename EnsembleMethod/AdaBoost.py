import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import DataPreproc


X_train, y_train, X_test, y_test = DataPreproc.ret_data()
n_estimators=30
min_error=1e-2
n_splits=5
range_estimators = 3


def random_weight(weight):
    ret = np.random.choice(np.array(range(weight.shape[0])), 
                           size=weight.shape[0], p=weight)
    return ret


def sample(weight, data, label):
    index_arr = random_weight(weight)
    return data[index_arr], label[index_arr]


def weight_error(y_true, y_pred, w):
    return np.sum(w * (y_true != y_pred))


def validation(clf, X, y, n_splits=n_splits):
    skf = StratifiedKFold(n_splits=n_splits)
    j = 0
    auc_0 = 0
    for train_idx, test_idx in skf.split(X,y):
        clf.fit(X[train_idx],y[train_idx])
        fpr, tpr, thresholds = roc_curve(y[test_idx], 
                                         clf.predict_proba(X[test_idx])[:,1])
        auc_1 = auc(fpr, tpr)
        auc_0 += auc_1
        print('Fold: %s, auc: %.3f' %(j+1, auc_1))
        j += 1
    auc_0 /= n_splits
    print('validation_auc:', auc_0)
    return auc_0


class myAdaBoost:
    
    def __init__(self, n_estimators=n_estimators, max_depth=1):
        self.h = {}    
        self.alpha = np.zeros(n_estimators)
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        for i in range(self.n_estimators):
            self.h[i]=DecisionTreeClassifier(max_depth=self.max_depth)
        
    def fit(self, X, y):
        D = np.full(X.shape[0], 1.0/X.shape[0])
        for i in range(self.n_estimators):
            print('round:',i)
            
            #根据分布重新采样X,并训练
            self.h[i].fit(*sample(D, X, y))
            
            #预测带权X，预测值为{0，1}
            y_pred = self.h[i].predict(X)
            
            #计算带权误差
            err = max(weight_error(y, y_pred, D),min_error)
            while(err>0.5):
                self.h[i].fit(*sample(D, X, y))
                y_pred = self.h[i].predict(X)
                err = max(weight_error(y, y_pred, D),min_error)
            print('err:',err)
     
            #根据误差计算当前基学习器的权重
            self.alpha[i] = 0.5*np.log((1.0-err)/err)
        
            #将预测值转换为{-1，1}
            y_pred_revise = (2 * y_pred - 1)
            y_train_revise = (2 * y - 1)
            
            #根据这次训练的结果更新分布D
            D *= np.exp(-self.alpha[i]*y_train_revise*y_pred_revise)
            D /= D.sum()
            
    def predict_proba(self, X):
        #根据训练好的强分类器预测测试集
        y_pred_prob = np.zeros((X.shape[0],2))      
        for i in range(self.n_estimators):
            y_pred_prob += self.alpha[i] * self.h[i].predict_proba(X)     
        y_pred_prob /= np.sum(self.alpha)
        return y_pred_prob


def main():
    best_ada = myAdaBoost(n_estimators = 1)
    best_valid = validation(best_ada, X_train, y_train)
    auc_list = []
    for i in range(range_estimators):
        tmp_ada = myAdaBoost(n_estimators = 10**i)
        tmp_validation = validation(tmp_ada, X_train, y_train)
        auc_list.append(tmp_validation)
        if tmp_validation > best_valid:
            best_valid = tmp_validation
            best_ada = tmp_ada
    
    best_ada.fit(X_train, y_train)    
    y_pred = best_ada.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    for i in range(range_estimators):
        print('n_estimators = %d, auc = %0.5f' % (10**i, auc_list[i]))
    print('best_ada.n_estimators:',best_ada.n_estimators)
    print('best_valid:',best_valid)

    #validation auc
    x = np.array(range(range_estimators))
    plt.plot(10**x, auc_list)
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Number of Estimators')
    plt.ylabel('Area Under Curve')
    plt.show()
    plt.close()

    #test auc
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    plt.plot(fpr, tpr, color='b',
             label=r'ROC (AUC = %0.5f)' % (roc_auc),
             lw=2, alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    plt.show()


if __name__ == '__main__':
    main()