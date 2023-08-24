#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


data=pd.read_csv('creditcard.csv',sep=',')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


data.describe()


# In[7]:


data.info()


# In[8]:


data.isnull().sum().any()


# In[9]:


class_count=pd.value_counts(data['Class'])
class_count.plot(kind='bar',rot=0)
plt.title("Transacton class distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[10]:


y = data['Class']
X = data.drop(['Class'], axis = 1)


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[13]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[14]:


confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)


# In[15]:


print("Confusion Matrix:\n", confusion)
print("\nClassification Report:\n", report)


# In[ ]:




