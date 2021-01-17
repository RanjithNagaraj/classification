#!/usr/bin/env python
# coding: utf-8

# # Text Classification - OneClass Classificaiton

# The one-class algorithms are based on recognition since their aim is to recognize data from a particular class, and reject data from all other classes. This is accomplished by creating a boundary that encompasses all the data belonging to the target class within itself, so when a new sample arrives the algorithm only has to check whether it lies within the boundary or outside and accordingly classify the sample as belonging to the target class or the outlier.

# Things we are going to discuss:
# 
# 1. Data Preparation 
# 2. Cleaning and Tokenization
# 3. Feature Extraction
# 4. Train One-class classificaiton model
# 5. Predict one-class on test data

# In[1]:


# Load packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.utils import shuffle
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem.porter import PorterStemmer
import string
import spacy
import nltk
from spacy.lang.en import English
#nltk_tokens = nltk.sent_tokenize(sentence_data)
#spacy.load('en')
#parser = English()


# In[2]:


# load dataset
bbc_df = pd.read_csv(r'C:\Users\ranjith_n\Downloads\prod_tasklist.csv',encoding='latin1')
bbc_df= bbc_df.head(200)
autoip = bbc_df[bbc_df['category'] == 1]['TASK']
noautoip = bbc_df[bbc_df['category'] == -1]['TASK']

bbc_df['category'] = bbc_df['category'].map({'automatable':1,'non-automatable':-1})


# In[18]:


# create a new dataset with only sport category data
sports_df = bbc_df[bbc_df['category'] == 1]


# In[19]:


sports_df.shape


# In[20]:


# create train and test data
train_text = sports_df['TASK'].tolist()
train_labels = sports_df['category'].tolist()

test_text = bbc_df['TASK'].tolist()
test_labels = bbc_df['category'].tolist()


# ## Data Cleaning and Tokenization

# In[21]:


import re
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
# stop words list
STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS)) 
# special characters
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”","''"]


# In[22]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 
# class for cleaning the text
class CleanTextTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
            return {}

def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text


# In[23]:


def tokenizeText(sample):
    
    tokens = sample.split()
    
    # lemmatization
    lemmas = []
    for tok in tokens:
        lemmas.append(lemmatizer.lemmatize(tok).strip())
    tokens = lemmas
    
    # remove stop words and special characters
    tokens = [tok for tok in tokens if tok.lower() not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    
    # only take words with length greater than or equal to 3
    tokens = [tok for tok in tokens if len(tok) >= 3]
    
    # remove remaining tokens that are not alphabetic
    tokens = [tok for tok in tokens if tok.isalpha()]
    
    # stemming of words
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]
    
    return list(set(tokens))



tokenizeText(train_text[9])



vectorizer = HashingVectorizer(n_features=20,tokenizer=tokenizeText)

features = vectorizer.fit_transform(train_text).toarray()
features.shape



from sklearn.calibration import CalibratedClassifierCV
# OneClassSVM algorithm
clf = OneClassSVM(nu=0.1, kernel="linear", gamma=0.01)
#kernel='rbf', nu=outlier_prop, gamma=0.000001
#scv_calibrated = CalibratedClassifierCV(clf)
pipe_clf = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('clf', clf)])


# In[27]:


# fit OneClassSVM model 
pipe_clf.fit(train_text, train_labels)











# let's predict the category of above random text
#print(pipe_clf.predict([test_text[3]]))
#print(pipe_clf.predict(['outlook failed to open because of new patch']))
#print(pipe_clf.predict(['laptop steal issue ']))
#print(pipe_clf.predict(['laptop not opened because of lockdown']))


