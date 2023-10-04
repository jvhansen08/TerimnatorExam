import pandas as pd
import numpy as np
import neattext.functions as nfx
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

def predict_emotion(text):
    print("I predict the emotion: " + pipe_lr.predict([text])[0])

if __name__ == '__main__':
    df = pd.read_csv('./archive/train.txt', sep=';', header=None, names=['Text', 'Emotion'])
    df['Clean_Text'] = df['Text'].apply(nfx.remove_stopwords)
    Xfeatures = df['Clean_Text']
    ylabels = df['Emotion']
    x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=42)
    pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('logisticRegression',LogisticRegression())])
    pipe_lr.fit(x_train,y_train)
    print("Type Ctrl-C to exit")
    while True:
        userText = input("Enter statement to predict emotion: ")
        predict_emotion(userText)

    

