
# coding: utf-8

# In[1]:

import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import time
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import nltk
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import sys


# In[11]:

def unique_list(l):
	ulist = []
	[ulist.append(x) for x in l if x not in ulist]
	return ulist


# In[2]:

data = pd.read_csv("C:\\Users\\KOGENTIX\\Desktop\\AC\\Reviews.csv",header=0)


# In[3]:

data_s = data[['id', 'categories', 'manufacturer','reviews.date','reviews.didPurchase',
               'reviews.doRecommend', 'reviews.rating', 'reviews.sourceURLs','reviews.text', 'reviews.title']]


# In[5]:

def split_to_test_train(rev_df):
	X_train, X_test, y_train, y_test = train_test_split(rev_df['reviews.text'],rev_df['sentiment'], test_size=0.2,stratify = rev_df['sentiment'])
	return (X_train, X_test, y_train, y_test)


# In[ ]:

def senti_tag(c):
    if c >=4.0:
        return 'pos'
    elif c >= 3.0 and c < 4.0:
        return 'neu'
    else:
        return 'neg'


# In[6]:

data_s['sentiment'] = data_s.apply(lambda x: senti_tag(x['reviews.rating']),axis=1)


# In[14]:

def filter_tokens(main_file_8):
	main_file_8=main_file_8.reset_index()
	print(main_file_8['index'].count())
	for each in range(main_file_8['index'].count()):
		#print(each)
		data = ''.join(str(main_file_8['reviews.text'][each]).lower()).strip()    
		ps = PorterStemmer()
		data = re.sub(r'[^a-zAA-Z0-9]',r' ',data) #html_parser.unescape(x).decode("utf8").encode('ascii','ignore')))
		data = re.sub(r'\w*\d\w*','',data).strip()
		data = ''.join([ps.stem(plural) for plural in data])
		data = ' '.join(unique_list([word for word,pos in nltk.pos_tag(data.split()) if pos == 'NNP' or pos == 'NN' or pos == 'VB' or pos == 'ADV' or pos == 'ADJ' ]))
		stop_words = set(stopwords.words('english'))
		word_tokens = data.split(" ")
		filtered_sentence = [w for w in word_tokens if not w in stop_words]
		filtered=' '.join(filtered_sentence).strip()
		for w in word_tokens:
			if w not in stop_words:
				filtered_sentence.append(w)
		shortword = re.compile(r'\W*\b\w{1,3}\b')
		data = shortword.sub('', filtered)
		main_file_8['reviews.text'][each] = data
	main_file_8 = main_file_8.dropna()
	return main_file_8


# In[ ]:

def model_cat(train_df,test_df):
	X_train = train_df['reviews']
	X_test = test_df['reviews']
	y_train =  train_df['label']
	y_test = test_df['label']
	main_file_df_bal = train_df
	print("train data after balance = ",main_file_df_bal.groupby('label').count())
	print(" ------- data_balancing ------- ")
	vectorizer1=weight_vectorizer(main_file_df_bal,cat)
	train_vectors = vectorizer1.transform(train_df['reviews'])
	y_train = main_file_df_bal['label']
	data_pos = test_df[test_df['label']==cat]
	data_neg = test_df[test_df['label']!=cat]
	data_pos['label']=1
	data_neg['label']=0
	data_test = data_pos.append(data_neg)
	test_vectors =  vectorizer1.transform(data_test['email'])
	y_test = data_test['label']
	model_classifier(train_vectors,y_train,cat,test_vectors,y_test,data_test)


# In[9]:

data_s['id'].count()


# In[ ]:

data_sw = filter_tokens(data_s)


# In[ ]:

data_sw['sentiment'] = data_sw.apply(lambda x: senti_tag(x['reviews.rating']),axis=1)


# In[ ]:

split_set=split_to_test_train(data_s)
X_train = split_set[0]
X_test = split_set[1]
y_train =  split_set[2]
y_test = split_set[3]


# In[ ]:

train_df = pd.DataFrame({'reviews':X_train.ravel(),'label':y_train.ravel()})
#print("train data before balance = ",train_df.groupby('label').count())
test_df = pd.DataFrame({'reviews':X_test.ravel(),'label':y_test.ravel()})


# In[ ]:

vectorizer1=weight_vectorizer(train_df)
train_vectors = vectorizer1.transform(main_file_df_bal['reviews'])

