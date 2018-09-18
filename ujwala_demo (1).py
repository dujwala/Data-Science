
# coding: utf-8

# In[27]:

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


# In[28]:

def unique_list(l):
	ulist = []
	[ulist.append(x) for x in l if x not in ulist]
	return ulist


# In[29]:

data = pd.read_csv("C:\\Users\\KOGENTIX\\Desktop\\AC\\Reviews.csv",header=0)


# In[30]:

data_s = data[['id','reviews.rating','reviews.text']]


# In[31]:

def split_to_test_train(rev_df):
	X_train, X_test, y_train, y_test = train_test_split(rev_df['reviews.text'],rev_df['sentiment'], test_size=0.2,stratify = rev_df['sentiment'])
	return (X_train, X_test, y_train, y_test)


# In[32]:

def senti_tag(c):
    if c >=4.0:
        return 'pos'
    elif c >= 3.0 and c < 4.0:
        return 'neu'
    else:
        return 'neg'


# In[33]:

data_s['sentiment'] = data_s.apply(lambda x: senti_tag(x['reviews.rating']),axis=1)


# In[36]:

data_sample=data_s.groupby('sentiment', group_keys=False).apply(lambda x: x.sample(min(len(x), 20)))


# In[24]:

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


# In[67]:

def model_cat(train_df,test_df):
	X_train = train_df['reviews']
	X_test = test_df['reviews']
	y_train =  train_df['label']
	y_test = test_df['label']
	main_file_df_bal = train_df
	print("train data after balance = ",main_file_df_bal.groupby('label').count())
	print(" ------- data_balancing ------- ")
	vectorizer1=weight_vectorizer(main_file_df_bal)
	train_vectors = vectorizer1.transform(train_df['reviews'])
	y_train = main_file_df_bal['label']
	test_vectors =  vectorizer1.transform(test_df['reviews'])
	y_test = test_df['label']
	model_classifier(train_vectors,y_train,test_vectors,y_test,test_df)


# In[71]:

def Classifier_Emails(train_vectors,y_train):
	#clf = svm.SVC(kernel='rbf',C=9000)
	svc=svm.SVC()
	clf_lr = LogisticRegressionCV(Cs=10, random_state=None)
	clf_lr.fit(train_vectors, y_train)
	pickle_fname = 'lr_lbfgs.sav'
	pickle.dump(clf_lr, open(pickle_fname, 'wb'))
	return clf_lr


# In[87]:

def model_classifier(train_vectors,y_train,test_vectors,y_test,main_file_test_df):
	from sklearn.decomposition import PCA
	from sklearn import cross_validation
	from sklearn.decomposition import TruncatedSVD
	clf = TruncatedSVD(800)
	#Xpca = clf.fit_transform(X)
	#pca = PCA(n_components=100)# adjust yourself
	#pca.fit(train_vectors)
	#clf.fit(train_vectors)
	#train_vectors_pca = clf.transform(train_vectors)
	model_built = RandomForestClassifier_Emails(train_vectors,y_train)
	results = model_built.predict(test_vectors)
	main_file_test_df['results']=results
	#main_file_test_df.to_csv("final_emails_test_15_dec_svm_"+str(cat)+".csv",sep=",", encoding='utf-8')
	list_ytest = list(y_test)
	list_res = list(results)
	ly=pd.DataFrame({'test':list_ytest,'result':list_res})
	ly.to_csv("C:\\Users\\KOGENTIX\\Desktop\\AC\\results_svm.csv",sep=",", encoding='utf-8')
	print("group by ly ",main_file_test_df.groupby('label').count())
	results = accuracy(list_ytest,list_res)
	print("results_svm acc "+" ",results)
	#results_auc=auc(list_ytest,list_res)
	#print("result_auc_svm "+" ",results_auc)
	return results


# In[57]:

data_s['id'].count()


# In[58]:

data_sw = filter_tokens(data_sample)


# In[13]:

data_s=pd.read_csv('C:\\Users\\KOGENTIX\\Desktop\\AC\\Reviews_filtered.csv',header=0)


# In[40]:

split_set=split_to_test_train(data_sw)
X_train = split_set[0]
X_test = split_set[1]
y_train =  split_set[2]
y_test = split_set[3]


# In[41]:

train_df = pd.DataFrame({'reviews':X_train.ravel(),'label':y_train.ravel()})
#print("train data before balance = ",train_df.groupby('label').count())
test_df = pd.DataFrame({'reviews':X_test.ravel(),'label':y_test.ravel()})


# In[44]:

def weight_vectorizer(main_file_df):
	X_train=main_file_df['reviews']
	y_train=main_file_df['label']
	from sklearn.feature_extraction import text
	#cus_stop_words = text.ENGLISH_STOP_WORDS
	vectorizer1 = TfidfVectorizer(min_df=4,
								max_df = 0.6,
								sublinear_tf = True,
								use_idf=True,
								ngram_range = (1,5))
	pickle_fname = 'vectorizer.sav'
	vectorizer1.fit(X_train)
	pickle.dump(vectorizer1, open(pickle_fname, 'wb'))
	with open(pickle_fname,'rb') as f:
		vectorizer1 = pickle.load(f)
	return vectorizer1


# In[45]:

vectorizer1=weight_vectorizer(train_df)
train_vectors = vectorizer1.transform(train_df['reviews'])
test_vectors = vectorizer1.transform(test_df['reviews'])


# In[52]:

def data_for_models(train_df,test_df):
	return model_cat(train_df,test_df)


# In[83]:

def RandomForestClassifier_Emails(train_vectors,y_train):
	rf = RandomForestClassifier(n_estimators=1000,
				min_impurity_split=1e-10)
	rf.fit(train_vectors, y_train)
	#pickle_fname_rf = 'rf.sav'
	#pickle.dump(rf, open(pickle_fname_rf, 'wb'))
	#with open(pickle_fname_rf,'rb') as f:
	#	rf = pickle.load(f)
	return rf


# In[88]:

data_for_models(train_df,test_df)


# In[73]:

def accuracy(list_ytest,list_res):
	count=0
	count_correct=0
	for i in range(len(list_ytest)):
		if list_res[i]==list_ytest[i]:
			count_correct=count_correct+1
		count = count+1
	return count_correct/count

def auc(list_ytest,list_res):
	from sklearn.metrics import roc_auc_score
	return roc_auc_score(list_ytest,list_res)


# In[ ]:



