#!/usr/bin/env python
# coding: utf-8

# # Publication Topic Analysis 
# 
# ## Objective
# ### A descriptive study of publications across OSU have been performed to understand the variety and depth of research across the university regarding Covid-19.
# 
# ## Data Source
# ### Elements Coronavirus Publications report
# 
# ## Scope of Analysis
# ### TimeFrame: 3/1/2020 - Present

# In[ ]:





# ### Importing required packages

# In[1]:


import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import cm
from mycolorpy import colorlist as mcp
import squarify
import requests
import io
import os


# ### Importing the base dataset

# In[2]:


df = pd.read_csv(r'./Data/Covid19publications_input_data.csv')


# ### Data Cleaning Operations
# #### a) Subset all publications after March 1st 2020 based on Publication date.
# #### b) The title and abstract column have been cleaned to remove punctuations and convert them to lowercase. The extra white spaces have also been removed.
# #### c) Author's department name has been combined with their First and Last Name.
# #### d) The topic and abstract column have been combined for performing topic modelling on it.

# In[3]:


df['publication-date'] = pd.to_datetime(df['publication-date'])
df = df[df['publication-date'] >= '3/1/2020']


# In[4]:


df['title_cleaned'] = df['title'].str.replace('[{}]'.format(string.punctuation), '').str.lower()
df['title_cleaned'] = df['title_cleaned'].str.strip()
df['Full Name'] = df['First Name'] +' '+df['Last Name']+ ' ( '+ df['Primary Group Descriptor']+' ) '

df_topic = df[['ID','Publication ID','title_cleaned','abstract']]
df_topic['abs_cleaned'] = df_topic['abstract'].str.replace('[{}]'.format(string.punctuation), '').str.lower()
df_topic['abs_cleaned'] = df_topic['abs_cleaned'].str.strip()
df_topic.abs_cleaned = df_topic.abs_cleaned.fillna('')
df_topic['topic_abstract'] = df_topic['title_cleaned']  + df_topic['abs_cleaned']


# ### Topic Modelling
# #### Algorithm: Latent Dirichlet Allocation (LDA): A statistical model which helps distribute the documents into a fixed number of topics.
# 
# #### Source: https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-latent-dirichlet-allocation-437c81220158

# In[ ]:





# ### Importing required packages

# In[5]:


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
#import pyLDAvis
#import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# ### Removing Stopwords

# In[6]:


# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# ### Preprocessing text for Modelling

# In[7]:


data = df_topic.topic_abstract.values.tolist()

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

#print(data_words[:1])


# ### Identifying phrases through bigrams and trigrams

# In[8]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=50) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=50)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# ### Defining functions for Text Cleaning

# In[9]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[10]:


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#print(data_lemmatized[:1])


# ### Creating the bag of words model to represent the text

# In[11]:


# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
#print(corpus[:1])


# In[12]:


# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


# ### Developing the LDA Model with the Bag of words representation
# 
# ### Model Parameters:
# #### corpus: Bag of words corpus
# #### id2word: Mapping of word-id to a token
# #### num_topics: Specified number of latent topics
# #### chunksize: Number of documents to be used in each training chunk
# #### update_every: Number of documents to be iterated through for each update
# #### passes: Number of passes through the corpus during training
# #### alpha: A-priori belief on document-topic distribution
# #### per_word_topics: the model computes a list of topics, sorted in descending order of most likely topics for each word
# 
# #### Model Information: https://radimrehurek.com/gensim/models/ldamodel.html

# In[13]:


# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=6, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# ### Identifying Top 10 words in each topic

# In[14]:


# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# ### Visual Representation of the topics

# In[15]:


import pyLDAvis.gensim_models
lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, sort_topics=False)
pyLDAvis.display(lda_display)


# ### Model Performance
# 
# #### Perplexity: Measurement of how well the model reproduces the held-out data 
# #### Coherence: Degree of semantic similarity between high scoring words in a topic

# In[16]:


# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# ### Model Interpretation
# 
# #### 6 topics have been created from the data based on topic interpretability and coherence score.
# #### Topic 1: covid , individual , case , state , vaccine , analysis , pandemic , patient , impact , symptom
# #### Topic 2: use , covid , news , trust , pandemic , prejudice , result , high , livestock , wellbeing
# #### Topic 3: find , firm , study , bocv , negative , however , pandemic , viral , calf , virus
# #### Topic 4: covid , discrimination , experience , use , pandemic , study , teacher , community , people , future
# #### Topic 5: covid , clinical , disease , sample , use , infection , study , pool , result , observe
# #### Topic 6: pandemic , covid , study , vaccine , state , report , mutation , gene , current , mask
