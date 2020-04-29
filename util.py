from sklearn.model_selection import cross_val_score

def train_evaluate_classfier(name, clf, x_train, y_train, x_test, y_test):
    print("===================== ", name, " =======================")
    # Train
    clf.fit(x_train, y_train)

    # evaulate accuracy on test set
    score = clf.score(x_test, y_test)
    print(name, " Final Test Score = ", score)

    # cross validation
    scores = cross_val_score(clf, x_train, y_train, cv=8)
    print(name, " Cross Validation Scores = ", scores)
    print(name, " Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("")