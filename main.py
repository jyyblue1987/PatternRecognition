from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

# Load the Pandas libraries with alias 'pd'
import pandas as pd
from util import *

# Load Train / Test Data
df = pd.read_csv(r'D_Train1.csv')
train_data = df.to_numpy()

X_Train = train_data[1:, 1:]
Y_Train = train_data[1:, [0]]
Y_Train = Y_Train.ravel()

scaler = preprocessing.StandardScaler().fit(X_Train)
# X_Train = scaler.transform(X_Train)
pca_dim = 2
pca = PCA(n_components=pca_dim)
pca.fit(X_Train)

print(pca.explained_variance_ratio_)

# X_Train = pca.transform(X_Train)

df = pd.read_csv(r'D_Test1.csv')

test_data = df.to_numpy()

X_Test = test_data[1:, 1:]
Y_Test = test_data[1:, [0]]
Y_Test = Y_Test.ravel()

# X_Test = pca.transform(X_Test)
# X_Test = scaler.transform(X_Test)

# ================= Navie Bayes ===============================
gnb = GaussianNB()

scores = cross_val_score(gnb, X_Train, Y_Train, cv=8)
print("Navie Bayes Cross Validation Scores = ", scores)
print("Navie Bayes Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

naive_model = gnb.fit(X_Train, Y_Train) # train
svm_score = gnb.score(X_Test, Y_Test)
print("Navie Bayes Final Test Score = ", svm_score)

# ================= SVM ===============================
# Train
clf = svm.SVC(decision_function_shape='ovr',kernel='linear', C=1)
scores = cross_val_score(clf, X_Train, Y_Train, cv=8)
print("SVM Cross Validation Scores = ", scores)
print("SVM Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Predict
clf.fit(X_Train, Y_Train)
svm_score = clf.score(X_Test, Y_Test)
print("SVM Final Test Score = ", svm_score)





