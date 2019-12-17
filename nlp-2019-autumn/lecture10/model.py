# encoding: utf8

import joblib
import xgboost as xgb
from sklearn import svm
from sklearn.model_selection import train_test_split

from data import build_dataset

DOC_EMBEDD = 'doc2vec.dv'
NEWS_DATASET = 'news.csv'

X, y = build_dataset(DOC_EMBEDD, NEWS_DATASET)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,)


# SVM Model
svm_clf = svm.SVC()
svm_clf.fit(X, y)
joblib.dump(svm_clf, 'svm.model')

svm_clf = joblib.load('svm.model')
svm_clf.predict(X_test)


# XGBOOST Model
xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train)

joblib.dump(xgb_clf, 'xgb.model')
xgb_clf.predict(X_test)
