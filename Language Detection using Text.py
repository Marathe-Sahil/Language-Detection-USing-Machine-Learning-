#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 


# In[4]:


data = pd.read_csv("language.csv")


# In[5]:


data


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer


# In[12]:


from sklearn.model_selection import train_test_split


# In[15]:


from sklearn.naive_bayes import MultinomialNB


# In[17]:


x = np.array(data['Text'])
y = np.array(data['language'])


# In[18]:


print(x)


# In[19]:


print(y)


# In[20]:


cv = CountVectorizer()
X = cv.fit_transform(x)


# In[22]:


X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.33 , random_state= 42)


# In[24]:


X_train


# In[25]:


print(X_train)


# In[39]:


print(y_train)


# In[40]:


model = MultinomialNB()


# In[41]:


model.fit(X_train,y_train)


# In[42]:


model.score(X_test,y_test)


# In[48]:


user = input(" enter the input ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)


# In[ ]:





# In[ ]:




