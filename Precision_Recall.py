import nltk
# import ssl
#
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download()
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
from ggplot import *
from sklearn import metrics
import ggplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
import pickle

def clean_text(text):
    if type(text) is str:
        text = text.lower()
        return ' '.join(tokenizer.tokenize(text))


realnews = pd.read_csv("realnews.csv")
realnews = realnews[["content", "title"]]
realnews['output'] = "real"
realnews = realnews.rename(columns={'content': 'text'})
fakenews = pd.read_csv("fake.csv")
fakenews = fakenews[fakenews["language"] == "english"]
fakenews = fakenews[["text", "title"]]
fakenews["output"] = "fake"
data = realnews.append(fakenews, ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)
tokenizer = RegexpTokenizer(r"\w+")
data["TEXT"] = [(clean_text(title)) for title in data["title"]]
new_df = data[data.TEXT.notnull()]
pickle.dump( new_df, open( "new_df", "wb" ) )

stopwords = nltk.corpus.stopwords.words("english")
vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize, stop_words=stopwords)
print(vectorizer)
# pickle.dump( vectorizer, open( "vectorizer", "wb" ) )

le = LabelEncoder()
X = vectorizer.fit_transform(new_df["TEXT"])
y = le.fit_transform(new_df["output"])

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)



#clf = svm.SVC(gamma=0.001, C=100.)
clf = OneVsRestClassifier(LogisticRegression())
# clf = RandomForestClassifier(n_estimators=100)

clf.fit(X_train, y_train)
kf = KFold(n_splits=3, shuffle=True)
scores = cross_val_score(clf, X, y, cv=kf)
print("Accuracy for 10 fold CV", scores)
print("Average accuracy: ", numpy.mean(scores))
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

preds = clf.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test, preds)

df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
#df.to_csv("/Users/shilpagundrathi/Downloads/RandomForest.csv")
auc = metrics.auc(fpr,tpr)
print(auc)

# g=  ggplot(df, aes(x='fpr', y='tpr')) +\
#     geom_abline(linetype='dashed')
# ggplot.ggsave(plot = g, filename = "new_test_file")

#print (ggplot(df, aes(x='fpr', y='tpr')) + \
    #eom_line(color='black') )