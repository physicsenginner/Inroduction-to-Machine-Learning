# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 20:19:43 2020

@author: Dogukan
"""


import  pandas as pd
# %% import data
data=pd.read_csv(r"twitter.csv",encoding="latin1")
data=pd.concat([data.gender,data.description],axis=1)
data.dropna(axis=0,inplace=True)
data.gender=[1 if i == "female" else 0 for i in data.gender]

# %% data cleaning 
# regular expression RE  mesela [^a-zA-Z]
import re
first_description=data.description[4]
description=re.sub("[^a-zA-Z]"," ",first_description) # a dan z ye A dan Z ye bulma geri kalanlari bosluk ile degis
description=description.lower()  # hepsi kucuk harf olur

# %% stopwords (irrelavent words) gereksiz kelimeler
import nltk # natural language tool lit
nltk.download("stopwords") # stopwords ler corpus klasorune indirilir
from nltk.corpus import stopwords 
nltk.download('punkt') # tokenize metodunu indirdi

#description=description.split()

# split yerine tokenizer kullanilabilir.

description=nltk.word_tokenize(description) # dont't - do ve not olarak ayrılır split metodunda ayrılmaz

# %% gereksiz kelimeleri çıkarma

description=[word for word in description if not word in set(stopwords.words("english"))]

# %% Lemmatization Loved => Love kelimenin kokunu bulur
import nltk as nlp
nltk.download('wordnet') # kok bulmak icin download ettik
lemma= nlp.WordNetLemmatizer()
description=[lemma.lemmatize(word) for word in description]

description=" ".join(description)


# %% 
description_list=[]
for description in data.description:
    description=re.sub("[^a-zA-Z]"," ",description) 
    description=description.lower()
    description=nltk.word_tokenize(description)
    #description=[word for word in description if not word in set(stopwords.words("english"))]
    lemma= nlp.WordNetLemmatizer()
    description=[lemma.lemmatize(word) for word in description]
    description=" ".join(description)
    description_list.append(description)

# %% bag of words
from  sklearn.feature_extraction.text import CountVectorizer # bag of words yaratmak için kullanilan method
max_features=5000

count_vectorizer = CountVectorizer(max_features=max_features,stop_words="english")
sparce_matrix=count_vectorizer.fit_transform(description_list).toarray() # x

print("en sik kullanilan {} kelimeler : {}".format(max_features,count_vectorizer.get_feature_names()))
# %% 
y=data.iloc[:,0].values # male or female classes
x=sparce_matrix
# train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=42)

# %% naive bayes

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)

# prediction
y_pred=nb.predict(x_test)

print("accuracy:",nb.score(y_pred.reshape(-1,1),y_test))



















