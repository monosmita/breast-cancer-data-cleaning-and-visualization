#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time 


# In[3]:


data=pd.read_csv("C:\\Users\\hp\\Downloads\\breast cancer data.csv")


# In[4]:


data.head()


# In[5]:


print(data.columns)


# In[6]:


y= data.diagnosis
drop_cols=['Unnamed: 32','id','diagnosis']
x=data.drop(drop_cols,axis=1) 
x.head()


# In[7]:


ax = sns.countplot(x=data.diagnosis,label = "Count")
B,M = y.value_counts()
print('number of bening tumors',B)
print('number of bening tumors',M)


# In[8]:


x.describe()


# In[8]:


data=x
data_std=(data-data.mean())/data.std()
data=pd.concat([y,data_std.iloc[:,0:10]],axis=1)
data=pd.melt(data,id_vars='diagnosis',
             var_name='features',
             value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x='features',y='value',hue='diagnosis',data=data,split=True,inner='quart')
plt.xticks(rotation=90);


# In[10]:


data=pd.concat([y,data_std.iloc[:,10:20]],axis=1)
data=pd.melt(data,id_vars='diagnosis',
             var_name='features',
             value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x='features',y='value',hue='diagnosis',data=data,split=True,inner='quart')
plt.xticks(rotation=90);


# In[11]:


data=pd.concat([y,data_std.iloc[:,20:30]],axis=1)
data=pd.melt(data,id_vars='diagnosis',
             var_name='features',
             value_name='value')
plt.figure(figsize=(20,20))
sns.violinplot(x='features',y='value',hue='diagnosis',data=data,split=True,inner='quart')
plt.xticks(rotation=90);


# In[12]:


plt.figure(figsize=(10,10))
sns.boxplot(x="features",y='value',hue='diagnosis',data=data)
plt.xticks(rotation=45);


# In[13]:


plt.figure(figsize=(15,10))
sns.jointplot(x=x.loc[:,'concavity_worst'],
             y=x.loc[:,'concave points_worst'],
             kind='reg',color='m')


# In[14]:


sns.set(style='whitegrid',palette='muted')
data=x
data_std=(data-data.mean())/data.std()
data=pd.concat([y,data_std.iloc[:,0:10]],axis=1)
data=pd.melt(data,id_vars='diagnosis',
             var_name='features',
             value_name='value')
plt.figure(figsize=(20,10))
sns.swarmplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=45);



# In[15]:


sns.set(style='whitegrid',palette='muted')
data=x
data_std=(data-data.mean())/data.std()
data=pd.concat([y,data_std.iloc[:,10:20]],axis=1)
data=pd.melt(data,id_vars='diagnosis',
             var_name='features',
             value_name='value')
plt.figure(figsize=(20,20))
sns.swarmplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=90);


# In[16]:


sns.set(style='whitegrid',palette='muted')
data=x
data_std=(data-data.mean())/data.std()
data=pd.concat([y,data_std.iloc[:,20:30]],axis=1)
data=pd.melt(data,id_vars='diagnosis',
             var_name='features',
             value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x='features',y='value',hue='diagnosis',data=data)
plt.xticks(rotation=90);


# In[24]:


f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(x.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax,cmap='PiYG');


# In[ ]:





# In[ ]:




