import numpy as np
from nltk.stem import PorterStemmer
import preprocessor as p
import string
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
import preprocessor as p



from textblob import Word

class CleanData:
    
    
    def CleanAll(data):
        cleaneddata=[]
        for sent in data:
            cleanedsent = CleanData.Clean(sent)
            if not cleanedsent:
                cleanedsent='null'
                print(sent +' (is none sent) and become '+cleanedsent)
            
            for word in cleanedsent.split(' '):
                if not word:
                    print(cleanedsent+' has empty word and was '+sent)
                    
                
                
            cleaneddata.append(cleanedsent)  
            
            
            
                

        return cleaneddata #cleaned Texts

    

    def Clean(sentence):
        #print('cleaning data')
        #sentence = p.clean(sentence)
        #lowercase
        sentence = sentence.lower()
        
        #sentence.encode('ascii', 'ignore').decode('ascii')
        
        ## removing remove hashtag, @user, link
        #sentence=' '.join(re.sub("(#[A-Za-z0-9]+)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",sentence).split())
        sentence=p.clean(sentence)
        
        #sentence=' '.join([i for i in sentence if not i.isdigit()])
        sentence = while_replace(sentence)
        sentence = clean_tweets(sentence)
        sentence = remove_punct(sentence)
        
        sentence = while_replace(sentence)
        
        
        
        #print('finish cleaning')
        return sentence#cleanedText
    
def while_replace(string):
    while '  ' in string:
        string = string.replace('  ', ' ')
        
    return string.strip()


def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

def clean_tweets(tweet):
 
    
#after tweepy preprocessing the colon symbol left remain after      #removing mentions
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
#replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
#remove emojis from tweet
    #tweet = emoji_pattern.sub(r'', tweet)
#filter using NLTK library append it to a string
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_tweet)
    #print(word_tokens)
    #print(filtered_sentence)return tweet
    

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
    
def cleanSample(text, stem=False): #clean text
    text = process_text(text)
    text = remove_content(text)
    text = " ".join([Word(word).lemmatize() for word in str(text).split()])
    return text

def cleanAllSample(data): #clean text
    data=data.apply(lambda x: process_text(x))
    data=data.apply(lambda x: remove_content(x))
    data=data.apply(lambda x: " ".join([Word(word).lemmatize() for word in str(x).split()]))
    return data