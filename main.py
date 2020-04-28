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

df = pd.read_csv(r'D_Test1.csv')

test_data = df.to_numpy()

X_Test = test_data[1:, 1:]
Y_Test = test_data[1:, [0]]
Y_Test = Y_Test.ravel()

predict = clf.predict(X_Test)

print(predict)
print(Y_Test)

zip_object = zip(predict, Y_Test)
error_count = 0
for i, j in zip_object:
    if i != j:
        error_count += 1

print("Test Count = ", len(Y_Test), "  Count = ", error_count)

