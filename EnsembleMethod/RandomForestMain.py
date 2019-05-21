import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import DataPreproc


X_train, y_train, X_test, y_test = DataPreproc.ret_data()
n_estimators=100
n_splits=5
max_estimators=100
range_estimators=10


def random_N(N):
    ret = np.random.randint(0,N,N)
    return ret


def sample(data, label):
    index_arr = random_N(data.shape[0])
    return data[index_arr], label[index_arr]


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


class myRandomForest:
    
    def __init__(self, n_estimators = n_estimators, max_features='log2'):
        self.n_estimators=n_estimators
        self.max_features=max_features
        self.h = {}  
        for i in range(self.n_estimators):
            self.h[i]=DecisionTreeClassifier(max_features=self.max_features)
     
    def fit(self, X, y):        
        for i in range(self.n_estimators):
            print('round:',i)
            X_resample, y_resample = sample(X, y)
            self.h[i].fit(X_resample,y_resample)
        
    def predict_proba(self, X):
        y_pred_prob = np.zeros((X.shape[0],2))    
        for i in range(self.n_estimators):
            y_pred_prob += self.h[i].predict_proba(X) 
        y_pred_prob /= self.n_estimators
        return y_pred_prob


def main():
    best_rf = myRandomForest(n_estimators = 1)
    best_valid = validation(best_rf, X_train, y_train)
    auc_list = {}
    for i in range(range_estimators):
        tmp_est = np.random.randint(1,max_estimators)
        tmp_rf = myRandomForest(n_estimators = tmp_est)
        tmp_validation = validation(tmp_rf, X_train, y_train)
        auc_list[tmp_est] = tmp_validation
        if tmp_validation > best_valid:
            best_valid = tmp_validation
            best_rf = tmp_rf
            
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
    auc_list = np.array(sorted(auc_list.items(), key = lambda item:item[0]))
    print(auc_list)
    print('best_rf.n_estimators:',best_rf.n_estimators)
    print('best_valid:',best_valid)
    
    #validation auc
    plt.plot(auc_list[:,0], auc_list[:,1])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Number of Estimators')
    plt.ylabel('Area Under Curve')
    plt.show()
    plt.close()
    
    #test_auc
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