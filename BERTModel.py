# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:42:59 2021

@author: SmaRt
"""

import numpy as np
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import Word

from tqdm import tqdm
import tensorflow as tf

from transformers import BertConfig,TFBertModel,BertTokenizer
from SupportClasses import CleanData

DSname='english_hasoc2019'
filename = 'TESTfile.csv'
#, nrows=50
df = pd.read_csv(filename, encoding='latin-1')

#df=df.rename(columns = {'Text':'text'})

df['Text']=CleanData.cleanAllSample(df['Text'])


nltk.download('punkt')



####### Change labels into 1=hate and 0=natural(no hate)
df.loc[(df.label == 1),'label']=2
df.loc[(df.label == 0),'label']=1
df.loc[(df.label == 2),'label']=0

############################
np.set_printoptions(suppress=True)
print(tf.__version__)
#############################

MODEL_TYPE = 'bert-base-uncased'
#############################
MAX_SEQUENCE_LENGTH = 200
#####################################


def _convert_to_transformer_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, str2, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1, str2,
            add_special_tokens=True,
            max_length=length,
            truncation_strategy=truncation_strategy)
        
        input_ids =  inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        
        return [input_ids, input_masks, input_segments]
    
    input_ids_q, input_masks_q, input_segments_q = return_id(
        title, None, 'longest_first', max_sequence_length)
    
    input_ids_a, input_masks_a, input_segments_a = return_id(
        '', None, 'longest_first', max_sequence_length)
        
    return [input_ids_q, input_masks_q, input_segments_q,
            input_ids_a, input_masks_a, input_segments_a]

def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.Text, instance.Text, instance.Text

        ids_q, masks_q, segments_q, ids_a, masks_a, segments_a = \
        _convert_to_transformer_inputs(t, q, a, tokenizer, max_sequence_length)
        
        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_segments_q.append(segments_q)
        input_ids_a.append(ids_a)
        input_masks_a.append(masks_a)
        input_segments_a.append(segments_a)
        
    return [np.asarray(input_ids_q, dtype=np.int32), 
            np.asarray(input_masks_q, dtype=np.int32), 
            np.asarray(input_segments_q, dtype=np.int32),
            np.asarray(input_ids_a, dtype=np.int32), 
            np.asarray(input_masks_a, dtype=np.int32), 
            np.asarray(input_segments_a, dtype=np.int32)]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


##################### load tokenizer
tokenizer = BertTokenizer.from_pretrained('_'+DSname+'_results/'+MODEL_TYPE+'_tokenizer/')
print('BertTokenizer Loaded')
####################################
output_categories = list(df.columns[[2]])
input_categories = list(df.columns[[1]])

test_inputs = compute_input_arrays(df, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_outputs = compute_output_arrays(df, output_categories)
test_outputs = np.asarray(test_outputs).astype(np.float32)

TARGET_COUNT = len(output_categories)

###################### load bert model
config = BertConfig()
config.output_hidden_states = False # Set to True to obtain hidden states


TFBmodel = TFBertModel.from_pretrained('bert-base-uncased', config=config)


###############
MAX_SEQUENCE_LENGTH = 200
output_categories = list(df.columns[[2]]) #classes column
TARGET_COUNT = len(output_categories)

q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
a_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
a_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
a_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
q_embedding = TFBmodel(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
a_embedding = TFBmodel(a_id, attention_mask=a_mask, token_type_ids=a_atn)[0]
q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
a = tf.keras.layers.GlobalAveragePooling1D()(a_embedding)
x = tf.keras.layers.Dropout(0.2)(q)

x = tf.keras.layers.Dense(TARGET_COUNT, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=[q_id, q_mask, q_atn, ], outputs=x)
model.load_weights('_'+DSname+'_results/'+MODEL_TYPE+'.h5')

print('Model Loaded')
#################################


y_preds_test=model.predict(test_inputs)
y_preds =np.round(y_preds_test)


"""## Evaluating the results LR model"""
from sklearn.metrics import classification_report
report = classification_report( test_outputs, y_preds )
from sklearn.metrics import confusion_matrix
print(report)
print(confusion_matrix(test_outputs, y_preds))


#################################
############## test on one sample
#Pridict classes: 1=hate and 0=natural(no hate)
#1st sample
Sample = "O racist, you are not ashamed of yourself!"
dfsample = pd.DataFrame(data={'Text':[Sample]})
dfsample['Text']=CleanData.cleanAllSample(dfsample['Text'])
input_categories_sample = list(dfsample.columns[[0]])
test_inputs_sample = compute_input_arrays(dfsample, input_categories_sample, tokenizer, MAX_SEQUENCE_LENGTH)

y_preds_test_sample=model.predict(test_inputs_sample)
y_preds_sample =np.round(y_preds_test_sample)
print('#################')
print('The prediction of the sample: {',Sample,'} is ',y_preds_sample[0][0])
print('#################')
########################
#2end sample
Sample = "What kind of human you are ?!"
dfsample = pd.DataFrame(data={'Text':[Sample]})
dfsample['Text']=CleanData.cleanAllSample(dfsample['Text'])
input_categories_sample = list(dfsample.columns[[0]])
test_inputs_sample = compute_input_arrays(dfsample, input_categories_sample, tokenizer, MAX_SEQUENCE_LENGTH)

y_preds_test_sample=model.predict(test_inputs_sample)
y_preds_sample =np.round(y_preds_test_sample)
print('#################')
print('The prediction of the sample: {',Sample,'} is ',y_preds_sample[0][0])
print('#################')


###########################  Riza's file sample
input_file = open('input_text.txt', 'r')
lines = input_file.readlines()
input_file.close()

sample = ''
for line in lines:
    sample = sample + line.strip()

print('INPUT TEXT: ' + sample)

dfsample = pd.DataFrame(data={'Text':[sample]})
dfsample['cleanedText']=CleanData.cleanAllSample(dfsample['Text'])

#Pridict classes: 1=hate and 0=natural(no hate)

y_preds_sample =np.round(model.predict(test_inputs_sample)) #BERT
if y_preds_sample[0][0] == 1:
    print('Input text contains hate')
else:
    print('Input text does not contain hate')

