from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

# Load the Pandas libraries with alias 'pd'
import pandas as pd
from util import *

# Load Train / Test Data
df = pd.read_csv(r'D_Train1.csv')
train_data = df.to_numpy()

X_Train = train_data[1:, 1:]
Y_Train = train_data[1:, [0]]
Y_Train = Y_Train.ravel()

pca_dim = 2
pca = PCA(n_components=pca_dim)
pca.fit(X_Train)

X_Train = pca.transform(X_Train)
print(pca.explained_variance_ratio_)

df = pd.read_csv(r'D_Test1.csv')

test_data = df.to_numpy()

X_Test = test_data[1:, 1:]
Y_Test = test_data[1:, [0]]
Y_Test = Y_Test.ravel()

X_Test = pca.transform(X_Test)

# ================= Navie Bayes ===============================
gnb = GaussianNB()
naive_model = gnb.fit(X_Train, Y_Train) # train
navie_y = naive_model.predict(X_Test)

# Evaluate Error Count
getErrorCount(navie_y, Y_Test)

# ================= SVM ===============================
# Train
clf = svm.SVC(decision_function_shape='ovr')
clf.fit(X_Train, Y_Train)

# Predict
svm_y = clf.predict(X_Test)

# Evaluate Error Count
getErrorCount(svm_y, Y_Test)





