from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Load the Pandas libraries with alias 'pd'
import pandas as pd
from util import *

# Load Train / Test Data
df = pd.read_csv(r'D_Train1.csv')
train_data = df.to_numpy()

X_Train = train_data[:, 1:]
Y_Train = train_data[:, [0]]
Y_Train = Y_Train.ravel()

f_dim = len(X_Train[0])

scaler = preprocessing.StandardScaler().fit(X_Train)


df = pd.read_csv(r'D_Test1.csv')

test_data = df.to_numpy()

X_Test = test_data[:, 1:]
Y_Test = test_data[:, [0]]
Y_Test = Y_Test.ravel()
names = ["Random", "Navie Bayer", "KNN", "SVM", "Gradient Descent"]

classifiers = [
    DummyClassifier(strategy="stratified"),
    GaussianNB(),
    KNeighborsClassifier(n_neighbors=4),
    svm.SVC(decision_function_shape='ovr',kernel='linear', C=1),
    SGDClassifier(loss="hinge", penalty="l2", max_iter=1000),
]

# iterate over classifiers with standard setting
score_list = []
for name, clf in zip(names, classifiers):
    score = train_evaluate_classfier(name, clf, X_Train, Y_Train, X_Test, Y_Test)
    score_list.append(score * 100)

x_pos = np.arange(len(names));
plt.bar(x_pos, score_list, align='center', alpha=0.5)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xticks(x_pos, names)
plt.ylabel('Accuracy')
plt.title('Classifiers Performance')
plt.show()

# Feature Reduction analysis for SVM classifier
print("================ Feature Reduction ================")
score_list = []
for pca_dim in range(2, f_dim):
    pca = PCA(n_components=pca_dim)
    pca.fit(X_Train)

    print("PCA Dimension = ", pca_dim, pca.explained_variance_ratio_)
    X_Train_Transform = pca.transform(X_Train)
    X_Test_Transform = pca.transform(X_Test)

    # svm
    clf = classifiers[3]
    score = train_evaluate_classfier("SVM: PCA Dim = " + str(pca_dim), clf, X_Train_Transform, Y_Train, X_Test_Transform, Y_Test)
    score_list.append(score * 100)

x_pos = np.arange(len(score_list))
plt.bar(x_pos, score_list, align='center', alpha=0.5)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xlabel('Dimension')
plt.ylabel('Accuracy')
plt.xticks(x_pos, range(2, f_dim))
plt.title('PCA Dimension vs Performance')
# plt.autoscale(axis='y',tight=True)
plt.show()