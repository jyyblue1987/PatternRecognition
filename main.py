from sklearn import svm
# Load the Pandas libraries with alias 'pd'
import pandas as pd
from numpy import *

df = pd.read_csv(r'D_Train1.csv')
train_data = df.to_numpy()

X_Train = train_data[1:, 1:]
Y_Train = train_data[1:, [0]]
Y_Train = Y_Train.ravel()

clf = svm.SVC(decision_function_shape='ovr')
clf.fit(X_Train, Y_Train)

# dec = clf.decision_function([[1]])
# print(dec.shape[1])

print(clf.predict([[-58,-58,-61,-59,-69,-81,-80]]))

df = pd.read_csv(r'D_Test1.csv')

test_data = df.to_numpy()

X_Test = test_data[1:, 1:]
Y_Test = test_data[1:, [0]]
Y_Test = Y_Test.ravel()

predict = clf.predict(X_Test)
print(predict)
print(Y_Test)