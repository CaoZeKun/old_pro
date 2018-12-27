import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

train_data1 = pd.read_csv('1all.csv')
#print(train_data1)
train_da_X = train_data1.iloc[:18,1:9]
train_da_Y = train_data1.iloc[:18,0]
test_da_X = train_data1.iloc[18:,1:9]
test_da_Y = train_data1.iloc[18:,0]
# print(type(test_da_Y))
# print(np.shape([test_da_Y]))
# from sklearn.tree import DecisionTreeClassifier
#
#
# from sklearn.neighbors import KNeighborsClassifier
#from sklearn.cross_validation import LeaveOneOut

# from sklearn.svm import SVC
#
# svc = SVC().fit(train_da_X,train_da_Y)
# predicted = svc.predict(test_da_X)
# value = abs(test_da_Y - predicted)/test_da_Y
# print(value)



# RandomForest
clf = RandomForestClassifier(n_estimators=10, random_state=123, min_samples_leaf=3).fit(train_da_X,train_da_Y)
# cv = LeaveOneOut()
# scores = cross_val_score(clf, train_da_X, train_da_Y,cv=cv)
# print(scores)
predicted = clf.predict(test_da_X)
print(test_da_Y)

print(predicted)
value = abs(test_da_Y - predicted)/test_da_Y
print(value)

# clf = RandomForestClassifier(n_estimators=1000, random_state=312, min_samples_leaf=3)
#scores = cross_val_score(clf, X, Y,cv=2,scoring='accuracy').mean()

# train_data2 = pd.read_csv('2.csv')
# train_da_X = train_data2.iloc[:3,1:10]
# train_da_Y = train_data2.iloc[:3,0]
# test_da_X = train_data2.iloc[2:,1:10]
# #test_da_X= test_da_X.reshape((1,-1))
# test_da_Y = train_data2.iloc[2:,0]