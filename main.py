from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

# Load the Pandas libraries with alias 'pd'
import pandas as pd
from util import *

# Load Train / Test Data
df = pd.read_csv(r'D_Train1.csv')
train_data = df.to_numpy()

X_Train = train_data[:, 1:]
Y_Train = train_data[:, [0]]
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

X_Test = test_data[:, 1:]
Y_Test = test_data[:, [0]]
Y_Test = Y_Test.ravel()

# X_Test = pca.transform(X_Test)
# X_Test = scaler.transform(X_Test)

# Random Classifier
dummy_clf = DummyClassifier(strategy="most_frequent")
train_evaluate_classfier("Random", dummy_clf, X_Train, Y_Train, X_Test, Y_Test)

# ================= Navie Bayes ===============================
gnb = GaussianNB()
train_evaluate_classfier("Navie Bayer", gnb, X_Train, Y_Train, X_Test, Y_Test)

# ================= KNN ===============================
neigh = KNeighborsClassifier(n_neighbors=4)
train_evaluate_classfier("KNN", neigh, X_Train, Y_Train, X_Test, Y_Test)

# ================= SVM ===============================
# Train
clf = svm.SVC(decision_function_shape='ovr',kernel='linear', C=1)
train_evaluate_classfier("SVM", clf, X_Train, Y_Train, X_Test, Y_Test)

# ================= SGD Classfier ===============================
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
train_evaluate_classfier("SGD", clf, X_Train, Y_Train, X_Test, Y_Test)



