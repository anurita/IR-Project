import nltk
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report


def clean_text(text):
    if type(text) is str:
        text = text.lower()
        return ' '.join(tokenizer.tokenize(text))
    
realnews = pd.read_csv("/Users/teju/Desktop/IR/proj/realnews.csv")
realnews = realnews[["content", "title"]]
realnews['output'] = "real"
realnews=realnews.rename(columns = {'content':'text'})

fakenews = pd.read_csv("/Users/teju/Desktop/IR/proj/fake.csv")
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
from sklearn import svm
from sklearn.svm import SVC


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(X_train, y_train)

from sklearn import metrics

y_pred = clf.predict(X_test)
score = metrics.accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)
from sklearn import metrics
pred = clf.predict_proba(X_test)[:,1]
fpr, tpr,_ = metrics.roc_curve(y_test, pred)
df = pd.Dataframe(dict(fpr=fpr, tpr = tpr))
df.to_csv("rf.csv")
