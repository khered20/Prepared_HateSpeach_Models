import numpy as np
import pickle
import nltk
import pandas as pd

from keras.models import load_model

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from SupportClasses import CleanData

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

#load tfidf
vectorizer = pickle.load(open("vectorizer_Davidson_hatespeech_twitter_data_2classes.pickle", 'rb'))
#load selected features
selectF = pickle.load(open("select_T.pickle", 'rb'))
#load SVM model
ML_model = pickle.load(open("Davidson_hatespeech_twitter_data_2classes_SVM_TFIDF.sav", 'rb'))
#Load CNN model
DL_model = load_model('Davidson_hatespeech_twitter_data_2classes_CNN_TFIDF.h5')

print('All Models Loaded')

input_file = open('input_text.txt', 'r')
lines = input_file.readlines()
input_file.close()

sample = ''
for line in lines:
    sample = sample + line.strip()

print('INPUT TEXT: ' + sample)

dfsample = pd.DataFrame(data={'Text':[sample]})
dfsample['cleanedText']=CleanData.cleanAllSample(dfsample['Text'])

#Extract tfidf vector from sample
tfidf = vectorizer.transform(dfsample['cleanedText']).toarray()

dfsample = pd.DataFrame(data={'Text':[sample]})
dfsample['cleanedText']=CleanData.cleanAllSample(dfsample['Text'])

#Extract tfidf vector from sample
tfidf = vectorizer.transform(dfsample['cleanedText']).toarray()

#Combine and select best features
Mft = np.concatenate([tfidf],axis=1)
X = selectF.transform(Mft)

#Pridict classes: 0=hate and 1=natural(no hate)
y_preds = ML_model.predict(X) #SVM
if y_preds[0] == 0:
    print('Input text contains hate')
else:
    print('Input text does not contain hate')


