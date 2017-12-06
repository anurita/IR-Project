# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:04:37 2017

@author: anu
"""

import numpy as np # linear algebra
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import re
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def removeStopWords(text):
    text_NoPunctuation = re.sub("[^a-zA-Z]", " ", text) 
    words = text_NoPunctuation.lower().split()
    englishStopWords = set(stopwords.words('english'))
    textremovedStopWords = [word for word in words if not word in englishStopWords]
    return( " ".join(textremovedStopWords))
    
def dataPreProcessing(dataFrame):
    dataFrame["text"].fillna(" ",inplace=True)    
    dataFrame["text"] = dataFrame["text"].apply(removeStopWords)
    dataFrame["title"].fillna(" ",inplace=True)    
    dataFrame["title"] = dataFrame["title"].apply(removeStopWords)
    return dataFrame
    
realnews = pd.read_csv("realnews.csv")
realnews = realnews[["content", "title"]]
realnews['output'] = "real"
realnews=realnews.rename(columns = {'content':'text'})

fakenews = pd.read_csv("fake.csv")
fakenews = fakenews[fakenews["language"] == "english"]
fakenews = fakenews[["text", "title"]]
fakenews["output"] = "fake"

data = realnews.append(fakenews, ignore_index=True)
data = dataPreProcessing(data)
#data = data.sample(frac=1).reset_index(drop=True)
tfidfVectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
clf = RandomForestClassifier(n_estimators=100)

"""X = tfidfVectorizer.fit_transform(data.text)
y = tfidfVectorizer.fit_transform(data.output)
clf.fit(X, y)
kf = KFold(n_splits=10, shuffle=True)
print(np.mean(cross_val_score(clf, X, y, cv=10)))

"""
train, test = train_test_split(data, test_size = 0.5)
train = dataPreProcessing(train)
test = dataPreProcessing(test)
y_train, y_test = train.output, test.output
X_train = tfidfVectorizer.fit_transform(train.title)
X_test = tfidfVectorizer.transform(test.title)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
score = metrics.accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)
print("classification report:")
print(metrics.classification_report(y_test, y_pred,target_names=train.output.unique()))
print("confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred))
