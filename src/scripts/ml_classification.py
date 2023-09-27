#!/usr/bin/env python
# coding: utf-8

# In[34]:


# sckit-learn
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, recall_score, precision_score


# Librosa
import librosa
import librosa.display

import IPython
import IPython.display as ipd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


# In[35]:


data = pd.read_csv('Data/features_3_sec.csv')
data.head()


# In[36]:


# treating any missing values
list(data.columns[data.isnull().any()]) # list is empty meaning there are no null values.


# In[81]:


# label encoding for models
encode = preprocessing.LabelEncoder()
data['label'] = encode.fit_transform(data['label'])

# independant and dependant variables with normalisation 
x = data.drop(['label', 'filename'], axis = 1)
y = data['label']
x = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(x), columns = x.columns)

# splitting data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# In[78]:


from sklearn import metrics
# function to build model
def buildModel(model):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print(metrics.classification_report(y_test, prediction, digits=4))
    metrics.plot_confusion_matrix(model, X_test, y_test)
    plt.show()

# function to initialise model
def createModel(model):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    
# logistic regression
buildModel(LogisticRegression())

# random forest classifier
buildModel(RandomForestClassifier())

# support vector machine
buildModel(SVC())


# In[83]:


# random forest classifier parameter tuning

# changing 'max_leaf_nodes' parameter
for i in range (2, 52, 2):
    rf = RandomForestClassifier(max_leaf_nodes = i)
    rf.fit(X_train, y_train)
    prediction = rf.predict(X_test)
    print(f'Maximum leaf nodes: {i} Accuracy {accuracy_score(y_test, prediction)}')
 

# changing 'max_depth' parameter
for i in range (1, 35):
    rf = RandomForestClassifier(max_depth = i)
    rf.fit(X_train, y_train)
    prediction = rf.predict(X_test)
    print(f'Maximum depth: {i} Accuracy {accuracy_score(y_test, prediction)}')


# In[79]:


from sklearn.model_selection import GridSearchCV

# support vector machine parameter tuning
parameters = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

search = GridSearchCV(SVC(), parameters, refit = True, verbose= 2)
search.fit(X_train, y_train)
print(search.best_estimator_)


# In[80]:


# final models 

buildModel(LogisticRegression()) # logistic regression

buildModel(RandomForestClassifier(max_depth = 25)) # random forest classifier

buildModel(SVC(C = 100, gamma = 1)) # support vector machine


# In[ ]:




