# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:42:59 2021

@author: SmaRt
"""

import numpy as np
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from textblob import Word
import pandas as pd




nltk.download('stopwords')
nltk.download('wordnet')


filename = 'TESTfile.csv'

#, nrows=50
df = pd.read_csv(filename, encoding='latin-1')
df=df.rename(columns = {'tweet':'Text'})
#lowercasing, stopwords, steming

import nltk
nltk.download('punkt')
def remove_content(text):
    text = re.sub(r"http\S+", "", text) #remove urls
    text=re.sub(r'\S+\.com\S+','',text) #remove urls
    text=re.sub(r'\@\w+','',text) #remove mentions
    text =re.sub(r'\#\w+','',text) #remove hashtags
    return text
def process_text(text, stem=False): #clean text
    text=remove_content(text)
    text = re.sub('[^A-Za-z]', ' ', text.lower()) #remove non-alphabets
    tokenized_text = word_tokenize(text) #tokenize
    stop =set( stopwords.words('english'))
    stopnew_stopwords = ['rt', 'amp', 'im','https', 'wasnt']
    stop = stop.union( stopnew_stopwords)
    clean_text = [
         word for word in tokenized_text
         if word not in stop
    ]
    if stem:
        clean_text=[stemmer.stem(word) for word in clean_text]
    return ' '.join(clean_text)
 

df['Text']=df['Text'].apply(lambda x: process_text(x))
df['Text']=df['Text'].apply(lambda x: remove_content(x))
df['Text']=df['Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in str(x).split()]))

dfclean = df.replace(r'^\s*$', np.nan, regex=True)
dfclean = dfclean.dropna()
dfclean.shape


vectorizer = pickle.load(open("vectorizer_Davidson_hatespeech_twitter_data_2classes.pickle", 'rb'))
tfidf = vectorizer.transform(dfclean['Text']).toarray()

import nltk
nltk.download('averaged_perceptron_tagger')



#tfidf = vectorizer.fit_transform(dfclean['Text']).toarray()
print('vectorizer_ was loaded from the folder')

loaded_model = pickle.load(open("Davidson_hatespeech_twitter_data_2classes_SVM_TFIDF.sav", 'rb'))

Mft = np.concatenate([tfidf],axis=1)
X = pd.DataFrame(Mft)
y = dfclean['label']
selectF = pickle.load(open("select_T.pickle", 'rb'))
X_cc = selectF.transform(Mft)
Encoder = LabelEncoder()
y = Encoder.fit_transform(y)

y_preds = loaded_model.predict(X_cc)
"""## Evaluating the results LR model"""
from sklearn.metrics import classification_report
report = classification_report( y, y_preds )
from sklearn.metrics import confusion_matrix
print(report)
print(confusion_matrix(y, y_preds))

#pickle.dump(select, open("select_T.pickle", "wb"))


# load model
from keras.models import load_model
modelDL = load_model('Davidson_hatespeech_twitter_data_2classes_CNN_TFIDF.h5')
y_preds = modelDL.predict_classes(X_cc)
"""## Evaluating the results LR model"""
from sklearn.metrics import classification_report
report = classification_report( y, y_preds )
from sklearn.metrics import confusion_matrix
print(report)
print(confusion_matrix(y, y_preds))   
    
    
