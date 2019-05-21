import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

train_data = np.loadtxt('train_set.txt')
test_data = np.loadtxt('test_set.txt')
X1 = train_data[:, :16]
y1 = train_data[:, 16]
X2 = test_data[:, :16]
y2 = test_data[:, 16]
clf = LogisticRegression(solver='newton-cg', multi_class='ovr').fit(X1,y1)
print("accuracy = %.4f" %(accuracy_score(y2, clf.predict(X2))))
print("micro Precision = %.4f" %(precision_score(y2, clf.predict(X2), average='micro')))
print("micro Recall = %.4f" %(recall_score(y2, clf.predict(X2), average='micro')))
print("micro f1 = %.4f" %(f1_score(y2, clf.predict(X2), average='micro')))
print("macro Precision = %.4f" %(precision_score(y2, clf.predict(X2), average='macro')))
print("macro Recall = %.4f" %(recall_score(y2, clf.predict(X2), average='macro')))
print("macro f1 = %.4f" %(f1_score(y2, clf.predict(X2), average='macro')))