# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 16:44:39 2021

"""



import numpy as np
import pandas as pd


from tqdm import tqdm
import tensorflow as tf

from transformers import BertConfig,TFBertModel,BertTokenizer
from transformers import XLNetConfig,TFXLNetModel,XLNetTokenizer
from transformers import RobertaConfig,TFRobertaModel,RobertaTokenizer
from transformers import XLMRobertaConfig,TFXLMRobertaModel,XLMRobertaTokenizer

from SupportClasses import CleanData

################################################
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-d', '-dataset', help='Specify the dataset used to train the model',
                    nargs='?', default='None', const='You should specify one of these hate speech datasets: hasoc2019, hasoc2020, default: -d hasoc2019')

parser.add_argument('-m', '-MODEL_TYPE', help='Specify model type, bert-base, bert-large, bert-m for multilingual, xlnet or roberta. default: -m bert-base-uncased',
                    nargs='?', default='None', const='bert-base-uncased')
parser.add_argument('-s', '-sample', help='Specify text sample or (txt-csv-tsv) file path',
                    nargs='?', default='None', const='You should specify a text sample for hate speech prediction, ex -s your_text')

args = parser.parse_args()

if args.m == 'None' and args.s == 'None' and args.d == 'None':
    print('~PreBERT.py -d hasoc2019 or hasoc2020')
    print('~PreBERT.py -m bert or xlnet -s specify a text or a file path in txt, csv or tsv formats')
    print('~model -d specify a the dataset used to train the model')
    print('~The model -m can be bert-base: bert-base-uncased or bert-large: bert-large-uncased or bert-m: bert-base-multilingual-cased')
    print('or xlnet: xlnet-base-cased or roberta: roberta-base or xlm-r: xlm-roberta')
    print('~If the path -s has txt format file, the system will check if the entire file contains hate')
    print('~If the format is csv or tsv, the system will check:')
    print('~If it has label column, the classification report will be printed.')
    print('~Otherwise, it will check all the samples and save the results in csv file.')
    
    
    
#DSname='specify model type such as hasoc2019, hasoc2020, #####'
#MODEL_TYPE = 'specify model type such as bert, xlnet, roberta'

#MODEL_TYPE = 'bert-base-uncased'
#DSname='english_hasoc2019'




if str(args.d).lower() in ['2019','hasoc2019','hasoc_2019', 'english_hasoc2019']:
    DSname='english_hasoc2019'
elif str(args.d).lower() in ['hasoc2020','hasoc_2020','hasoc_2020_en','hasoc_2020_en_train']:
    DSname='hasoc_2020_en_train'
elif str(args.d).lower() in ['bert-m','bert-base-m', 'bert-base-multilingual' 'bert-base-multilingual-cased']:
    DSname='english_hasoc2019'
else:
    DSname=args.d

if str(args.m).lower() in ['bert','bert-base', 'bert-base-uncased']:
    MODEL_TYPE = 'bert-base-uncased'
elif str(args.m).lower() in ['bert-base','bert-large-uncased']:
    MODEL_TYPE = 'bert-large-uncased'
elif str(args.m).lower() in ['bert-m','bert-base-m', 'bert-base-multilingual' 'bert-base-multilingual-cased']:
    MODEL_TYPE = 'bert-base-multilingual-cased'
elif str(args.m).lower() in ['xlnet','xlnet-base', 'xlnet-base-cased']:
    MODEL_TYPE = 'xlnet-base-cased'
elif str(args.m).lower() in ['roberta','roberta-base']:
    MODEL_TYPE = 'roberta-base'
elif str(args.m).lower() in ['xlm-r','xlm-roberta', 'xlm-roberta-base']:
    MODEL_TYPE = 'xlm-roberta-base'
else:
    MODEL_TYPE = args.m

if DSname == 'None':
    DSname='english_hasoc2019'
if MODEL_TYPE == 'None':
    MODEL_TYPE = 'bert-base-uncased'
      



############################
MAX_SEQUENCE_LENGTH = 200

############################
np.set_printoptions(suppress=True)
print(tf.__version__)

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

def compute_input_arrays_roberta(df, columns, tokenizer, max_sequence_length):
    input_ids_q, input_masks_q, input_segments_q = [], [], []
    input_ids_a, input_masks_a, input_segments_a = [], [], []
    for _, instance in tqdm(df[columns].iterrows()):
        t, q, a = instance.text, instance.text, instance.text

        ids_q, masks_q, ids_a, masks_a = \
        _convert_to_transformer_inputs(t, q, a, tokenizer, max_sequence_length)
        
        input_ids_q.append(ids_q)
        input_masks_q.append(masks_q)
        input_ids_a.append(ids_a)
        input_masks_a.append(masks_a)
        
    return [np.asarray(input_ids_q, dtype=np.int32), 
            np.asarray(input_masks_q, dtype=np.int32), 
            np.asarray(input_ids_a, dtype=np.int32), 
            np.asarray(input_masks_a, dtype=np.int32)
            ]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns])


##################### load tokenizer and bert model
if MODEL_TYPE == 'bert-base-uncased':
    config = BertConfig()
    config.output_hidden_states = False # Set to True to obtain hidden states
    tokenizer = BertTokenizer.from_pretrained('_'+DSname+'_results/'+MODEL_TYPE+'_tokenizer/')
    TFBmodel = TFBertModel.from_pretrained(MODEL_TYPE, config=config)
    
elif MODEL_TYPE == 'bert-large-uncased' or MODEL_TYPE == 'bert-base-multilingual-cased':
    config = BertConfig()
    config.output_hidden_states = False # Set to True to obtain hidden states
    tokenizer = BertTokenizer.from_pretrained('_'+DSname+'_results/'+MODEL_TYPE+'_tokenizer/')
    bert_model = TFBertModel.from_pretrained(MODEL_TYPE)
    
elif MODEL_TYPE == 'xlnet-base-cased':
    config = XLNetConfig()
    config.output_hidden_states = False # Set to True to obtain hidden states
    tokenizer = XLNetTokenizer.from_pretrained('_'+DSname+'_results/'+MODEL_TYPE+'_tokenizer/')
    TFBmodel = TFXLNetModel.from_pretrained(MODEL_TYPE)
elif MODEL_TYPE == "roberta-base":
    #jplu/tf-xlm-roberta-base
    config = RobertaConfig()
    config.output_hidden_states = False # Set to True to obtain hidden states
    tokenizer = RobertaTokenizer.from_pretrained('_'+DSname+'_results/'+MODEL_TYPE+'_tokenizer/')
    TFBmodel = TFRobertaModel.from_pretrained(MODEL_TYPE)
elif MODEL_TYPE == 'xlm-roberta-base':
    #jplu/tf-xlm-roberta-base
    config = XLMRobertaConfig()
    config.output_hidden_states = False # Set to True to obtain hidden states
    tokenizer = XLMRobertaTokenizer.from_pretrained('_'+DSname+'_results/'+MODEL_TYPE+'_tokenizer/')
    TFBmodel = TFXLMRobertaModel.from_pretrained(MODEL_TYPE)


print('BertTokenizer Loaded')




TARGET_COUNT = 1

###################### load bert model
###############


q_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
a_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
q_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
a_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
q_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
a_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)

if MODEL_TYPE == 'roberta-base' or MODEL_TYPE == 'xlm-roberta-base':
    
    q_embedding = TFBmodel(q_id, attention_mask=q_mask)[0]
    a_embedding = TFBmodel(a_id, attention_mask=a_mask)[0]
    q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
    a = tf.keras.layers.GlobalAveragePooling1D()(a_embedding)
    x = tf.keras.layers.Dropout(0.2)(q)
    
    x = tf.keras.layers.Dense(TARGET_COUNT, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=[q_id, q_mask, a_id, a_mask ], outputs=x)
else:
    q_embedding = TFBmodel(q_id, attention_mask=q_mask, token_type_ids=q_atn)[0]
    a_embedding = TFBmodel(a_id, attention_mask=a_mask, token_type_ids=a_atn)[0]
    q = tf.keras.layers.GlobalAveragePooling1D()(q_embedding)
    a = tf.keras.layers.GlobalAveragePooling1D()(a_embedding)
    x = tf.keras.layers.Dropout(0.2)(q)
    
    x = tf.keras.layers.Dense(TARGET_COUNT, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=[q_id, q_mask, q_atn,a_id, a_mask, a_atn ], outputs=x)
model.load_weights('_'+DSname+'_results/'+MODEL_TYPE+'.h5')

print(MODEL_TYPE+' Model Loaded')

print(DSname)
print(MODEL_TYPE)

import timeit
start = timeit.default_timer()




sample = args.s.lower()
if len(sample)>3:
    format_file=sample[len(sample)-4:len(sample)]
else:
    format_file='none'
##################################


#sample='input_text.txt'
#MODEL_TYPE = 'xlnet-base-cased'
format_file=sample[len(sample)-4:len(sample)]
if format_file == '.txt':
    input_file = open('input_text.txt', 'r')
    lines = input_file.readlines()
    input_file.close()
    sample = ''
    for line in lines:
        sample = sample + line.strip()
    print('INPUT TEXT: ' + sample)
    df = pd.DataFrame(data={'Text':[sample]})
elif format_file == '.tsv':
    df = pd.read_csv(sample, sep='\t')
elif format_file == '.csv':
    df = pd.read_csv(sample)
else:
    df = pd.DataFrame(data={'Text':[sample]})
    

df=df.rename(columns = {'task_1':'label'})
df=df.rename(columns = {'LABEL':'label'})
df=df.rename(columns = {'class':'label'})

df=df.rename(columns = {'text':'Text'})
df=df.rename(columns = {'tweet':'Text'})


############################
df['Text']=CleanData.cleanAllSample(df['Text'])

#####################################
input_categories = ['Text']
if MODEL_TYPE == 'roberta-base' or MODEL_TYPE == 'xlm-roberta-base':
    test_inputs = compute_input_arrays_roberta(df, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
else:
    test_inputs = compute_input_arrays(df, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
#####################################
if 'label' in df.columns:
    #df.loc[(df.label in ['NOT','not','false','FALSE','no-hate']),'label']=0
    #df.loc[(df.label in ['HOF','OFF','true','TRUE','hate']),'label']=1
    df.loc[(df.label == 'NOT'),'label']=0
    df.loc[(df.label == 'not'),'label']=0
    df.loc[(df.label == 'false'),'label']=0
    df.loc[(df.label == 'no-hate'),'label']=0
    
    df.loc[(df.label == 'HOF'),'label']=1
    df.loc[(df.label == 'OFF'),'label']=1
    df.loc[(df.label == 'true'),'label']=1
    df.loc[(df.label == 'hate'),'label']=1
    
    # output_categories = list(df.columns[[2]])
    # input_categories = list(df.columns[[1]])
    output_categories = ['label']
    test_outputs = compute_output_arrays(df, output_categories)
    # from sklearn.preprocessing import LabelEncoder
    # Encoder = LabelEncoder()
    # y = Encoder.fit_transform(test_outputs)
    #test_outputs=Encoder.inverse_transform(y)
    test_outputs = np.asarray(test_outputs).astype(np.float32)
    #TARGET_COUNT = len(output_categories)

    y_preds_test=model.predict(test_inputs)
    y_preds =np.round(y_preds_test)
    
    
    """## Evaluating the results LR model"""
    from sklearn.metrics import classification_report
    report = classification_report( test_outputs, y_preds )
    from sklearn.metrics import confusion_matrix
    print(report)
    print(confusion_matrix(test_outputs, y_preds))
    
else:

    y_preds_sample =np.round(model.predict(test_inputs)) #BERT
    if len(df) == 1:
        
        if y_preds_sample[0][0] == 1:
            print('Input text in ('+sample+') contains hate')
        else:
            print('Input text in ('+sample+') does not contain hate')
    else:
        df['prediction']=y_preds_sample
        df.loc[(df.prediction == 0),'prediction']='does not contain hate'
        df.loc[(df.prediction == 1),'prediction']='contains hate'
        print(df)
        save_path='hate_prediction_'+sample[0:len(sample)-4]+'.csv'
        df.to_csv(save_path)
        print('The prediction resuts are saved in '+save_path)
        
      
stop = timeit.default_timer()
runingtime=stop - start
print('The running time in sec: ', round(runingtime, 3))