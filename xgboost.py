import os, pickle
import numpy as np
import pandas as pd
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler

    

def train_and_test_xgboost():
    X, y = make_hastie_10_2(random_state=0)
    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]

    print(X_train.shape)
    print(y_train.shape)

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
        max_depth=1, random_state=0).fit(X_train, y_train)
    print(y_test)
    clf.score(X_test, y_test)

    with open('xgb.pkl', 'wb') as f:
        pickle.dump(clf, f)

    return clf


def infer_xgboost(clf, input_array):

    print(clf.predict(input_array))
    return


if(__name__=="__main__"):
    clf = train_and_test_xgboost()
    test_input = np.array([0 for i in range(10)])
    test_input = test_input.reshape(1,-1)
    print(test_input.shape)
    infer_xgboost(clf, test_input)
