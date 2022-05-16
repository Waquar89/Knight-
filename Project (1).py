#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('iris.csv')


# In[4]:


df.head()


# In[5]:


from sklearn.datasets import load_iris
iris = load_iris()
print(iris.DESCR)


# In[6]:


features = iris['data']


# In[7]:


feature_names = iris['feature_names']


# In[8]:


features.shape


# In[9]:


feature_names


# In[10]:


df.isnull().sum()


# In[11]:


X=df.iloc[:, :-1]
y=df.iloc[:, -1]


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101 )


# In[14]:


print(X.shape, X_train.shape, X_test.shape)


# In[15]:


X_train


# In[16]:


X_test


# In[17]:


y_train


# In[18]:


y_test


# In[19]:


from sklearn.linear_model import LogisticRegression


# In[20]:


mod_reg = LogisticRegression()


# In[21]:


mod_reg.fit(X_train, y_train)


# In[22]:


print("Accuracy:", mod_reg.score(X_test, y_test)*100)


# In[ ]:




