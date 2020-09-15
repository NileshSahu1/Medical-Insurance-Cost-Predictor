#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns


# In[2]:


#importing dataset
data=pd.read_csv('insurance.csv')
data.head()


# In[3]:


#before appyling machine learning algorithms. we have to convert categorical data into numerical data


# In[4]:


data['region'] = data['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)


# In[5]:


X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


# In[6]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])


# In[7]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 4] = le.fit_transform(X[:, 4])


# In[8]:


X


# In[13]:


# splitting data set and training set
from sklearn.model_selection import train_test_split
X_test,X_train,y_test,y_train=train_test_split(X,y,test_size=0.2,random_state=0)


# In[14]:


#Linear regression 
from sklearn.linear_model import LinearRegression
regressor1=LinearRegression()
regressor1.fit(X_train,y_train)


# In[19]:


y_pred1=regressor1.predict(X_test)


# In[24]:


from sklearn.metrics import r2_score
r2_score(y_pred1,y_test)


# In[ ]:


# so the accuracy is very low. lets try and apply different algorithms 


# In[38]:


# Decision tree regression
from sklearn.tree import DecisionTreeRegressor
regressor2=DecisionTreeRegressor(random_state=0)
regressor2.fit(X,y)


# In[39]:


y_pred2=regressor2.predict(X_test)


# In[40]:


from sklearn.metrics import r2_score
r2_score(y_pred2,y_test)


# In[ ]:


# very impressive accuracy score of 99.82. lets try one more algorithm.


# In[41]:


from sklearn.ensemble import RandomForestRegressor
regressor3=RandomForestRegressor(n_estimators=10,random_state=0)
regressor3.fit(X,y)


# In[42]:


y_pred3=regressor3.predict(X_test)


# In[43]:


from sklearn.metrics import r2_score
r2_score(y_pred3,y_test)


# In[ ]:


# 96.14 good accuracy


# In[ ]:


# so the accuracy of Linear regressor is 67.75.
# so the accuracy of DecisionTree regressor is 99.82.
# so the accuracy of RandomForest regressor is 96.14
#DecisionTree regressor won. so we will use DecisionTree regressor


# In[44]:


# Saving model to disk
pickle.dump(regressor2, open('model.pkl','wb'))


# In[46]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[19, 0, 33, 3, 1,2]]))


# In[ ]:




