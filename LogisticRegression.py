# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:48:53 2017

@author: ashwini
"""
import nltk
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import numpy
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report


def clean_text(text):
    if type(text) is str:
        text = text.lower()
        return ' '.join(tokenizer.tokenize(text))
    
realnews = pd.read_csv("C:\\Users\\realnews.csv")
realnews = realnews[["content", "title"]]
realnews['output'] = "real"
realnews=realnews.rename(columns = {'content':'text'})

fakenews = pd.read_csv("C:\\Users\\fake.csv")
fakenews = fakenews[fakenews["language"] == "english"]
fakenews = fakenews[["text", "title"]]
fakenews["output"] = "fake"

data = realnews.append(fakenews, ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)

tokenizer = RegexpTokenizer(r"\w+")

    
data["TEXT"] = [(clean_text(title)) for title in data["title"]]
new_df=data[data.TEXT.notnull()] 

stopwords = nltk.corpus.stopwords.words("english")

vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize, stop_words=stopwords)
le = LabelEncoder()

X = vectorizer.fit_transform(new_df["TEXT"])
y = le.fit_transform(new_df["output"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)

kf = KFold(n_splits=10, shuffle=True)

scores = cross_val_score(clf, X, y, cv=kf)

print("Accuracy for 10 fold CV", scores)
print("Average accuracy: ", numpy.mean(scores))
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le.classes_))
