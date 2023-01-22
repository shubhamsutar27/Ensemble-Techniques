#!/usr/bin/env python
# coding: utf-8

# # Light Gradient Boosting Machine

# In[1]:


pip install lightgbm


# In[2]:


import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv('SVMtrain.csv')
data.head()


# In[4]:


data.drop(['Sex'],axis=1,inplace=True)
data.head()


# In[5]:


data.info()


# In[6]:


x = data.drop(['Embarked','PassengerId'],axis=1)
y = data['Embarked']


# In[7]:


data['Embarked'].value_counts()


# In[8]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)


# In[9]:


model = lgb.LGBMClassifier(random_state=42)
model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)],verbose=20)


# In[10]:


print('Training accuracy {:.4f}'.format(model.score(x_train,y_train)))
print('Testing accuracy {:.4f}'.format(model.score(x_test,y_test)))


# In[11]:


lgb.plot_importance(model)


# In[14]:


lgb.plot_metric(model)


# In[15]:


metrics.plot_confusion_matrix(model,x_test,y_test,cmap='Blues_r')


# In[16]:


print(metrics.classification_report(y_test,model.predict(x_test)))


# # Gradient Boosting

# In[17]:


from sklearn.ensemble import GradientBoostingClassifier


# In[18]:


model_1 = GradientBoostingClassifier().fit(x_train,y_train)


# In[19]:


print(metrics.classification_report(y_test,model_1.predict(x_test)))


# In[ ]:




