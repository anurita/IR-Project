from textblob import TextBlob
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
from nltk.sentiment import SentimentIntensityAnalyzer
import operator

def clean_text(text):
    if type(text) is str:
        text = text.lower()
        return ' '.join(tokenizer.tokenize(text))
realnews = pd.read_csv("realnews.csv")
realnews = realnews[["content", "title"]]
realnews['output'] = "real"
realnews=realnews.rename(columns = {'content':'text'})

fakenews = pd.read_csv("fake.csv")
fakenews = fakenews[fakenews["language"] == "english"]
fakenews = fakenews[["text", "title"]]
fakenews["output"] = "fake"

data = realnews.append(fakenews, ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)
tokenizer = RegexpTokenizer(r"\w+")

data["TEXT"] = [(clean_text(text)) for text in data["text"]]
new_df=data[data.TEXT.notnull()] 
#new_df=data[data.title.notnull()] 

sentiment = SentimentIntensityAnalyzer()
polarities=[]

polarity_scores = []

for text in new_df['TEXT']:
    polarity_scores.append(TextBlob(text).sentiment.polarity)
    scores = sentiment.polarity_scores(text)
    subset = {k: scores[k] for k in ('neg', 'neu', 'pos')}
    t = max(subset.items(), key=operator.itemgetter(1))[0]
    polarities.append(t)

new_df["polarities"] = polarities
new_df.head()
