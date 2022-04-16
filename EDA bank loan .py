#!/usr/bin/env python
# coding: utf-8

# In[118]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')

#setting up plot style
style.use('seaborn-poster')
style.use('fivethirtyeight')


# In[119]:


import warnings
warnings.filterwarnings('ignore')


# In[120]:


#set_option( ) is used to adjust the jupiter view
pd.set_option('display.max_rows',500)
pd.set_option('display.max_rows',500)
pd.set_option('display.width',1000)
pd.set_option('display.expand_frame_repr',False)


# In[121]:


application=pd.read_csv(r'C:\Users\abhir\Downloads\28th\28th\application_data.csv')
previous=pd.read_csv(r'C:\Users\abhir\Downloads\archive (1)\previous_application.csv')


# In[122]:


application.head()


# In[123]:


previous.head()


# In[124]:


#database dimensions
print('database dimensions',application.shape)
print('previous',previous.shape)


# In[125]:


#database size
print('database size',application.size)
print('database size',previous.size)


# In[126]:


#for large dataframes use verbose 
application.info(verbose=True)


# In[127]:


previous.info(verbose=True)


# In[128]:


application.describe()


# In[129]:


previous.describe()


# In[130]:


import missingno as mn
mn.matrix(application)


# In[131]:


#sum of null value in each column
application.isnull().sum()


# In[132]:


application.shape[0]


# In[133]:


# % of null values
round(application.isnull().sum()/application.shape[0]*100.00,2)


# In[134]:


#to plot the columns Vs missing value % and taking 40% as cutoff mark
null_application=pd.DataFrame(application.isnull().sum()/application.shape[0]*100).reset_index()
null_application.columns=['Column name','Null Value percentage']
fig=plt.figure(figsize=(16,8))
ax=sns.pointplot(x='Column name',y='Null Value percentage',data=null_application,color='green')
plt.xticks(rotation=90,fontsize=7)
ax.axhline(40,ls='--',color='green')
plt.title('percentage of missing values')
plt.xlabel=('columns')
plt.ylabel=('Null values %')
plt.show()


# In[135]:


#columns having more than 40% empty rows
nullColumns_40=null_application[null_application['Null Value percentage']>=40]
nullColumns_40


# In[136]:


#no of coluns with missing values more than or equal to 40%
len(nullColumns_40)


# In[137]:


mn.matrix(previous)


# In[138]:


#number of null values in each column in previous dataset
previous.isnull().sum()


# In[139]:


# % of null values in each columns
round(previous.isnull().sum()/previous.shape[0]*100.00,2)


# In[140]:


#to plot the columns Vs missing value % and taking 40% as cutoff mark
null_previous=pd.DataFrame((previous.isnull().sum())*100/previous.shape[0]).reset_index()
null_previous.columns=['Column name','Null Value percentage']
fig=plt.figure(figsize=(16,8))
ax=sns.pointplot(x='Column name',y='Null Value percentage',data=null_previous,color='green')
plt.xticks(rotation=90,fontsize=7)
ax.axhline(40,ls='--',color='green')
plt.title('percentage of missing values')
plt.xlabel=('columns')
plt.ylabel=('Null values %')
plt.show()


# In[142]:


nullColumns_40prev=null_previous[null_previous['Null Values Percentage']>=40]
nullColumns_40prev


# In[ ]:


#verfying whether EXT_SOURCE is correlated with target
plt.figure(figsize=(8,8))
source_corr=application[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','TARGET']]
ax=sns.heatmap(source_corr.corr(),xticklabels=source_corr.columns, yticklabels=source_corr.columns,cmap='cubehelix',annot=True,linewidth=1)


# In[ ]:


#creating a list with columns having >40% null values and adding the EXT_SOURCE
#.list() is used to convert a series to list
unwanted_columns=nullColumns_40['Column name'].tolist()+['EXT_SOURCE_2','EXT_SOURCE_3']
len(unwanted_columns)


# In[ ]:


#finding correlation for flag documents
flag_corr=application[[ 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7', 
           'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
           'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
           'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21','TARGET']]
ay=sns.heatmap(flag_corr.corr(),cmap='cubehelix')


# In[ ]:


#added this as I got error 'str' object is not callable
import matplotlib.pyplot as plt
from importlib import reload
plt=reload(plt)


# In[ ]:


#checking wether the submitted flag_docments is related with loan repayment status
col_Doc = [ 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7', 
           'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
           'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
           'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
df_flag = application[col_Doc+["TARGET"]]
df_flag["TARGET"] = df_flag["TARGET"].replace({1:"Defaulter",0:"Repayer"})
fig = plt.figure(figsize=(21,24))
j=0
for i in col_Doc:
    plt.subplot(5,4,j+1)
    ax = sns.countplot(df_flag[i],hue=df_flag["TARGET"],palette=["r","g"])
    plt.yticks(fontsize=0.1)
    plt.title(i)
    plt.yticks(fontsize=8)
    plt.xlabel("")
    plt.ylabel("")
    j=j+1


# In[ ]:


col_Doc.remove('FLAG_DOCUMENT_3') 
unwanted_columns = unwanted_columns + col_Doc
len(unwanted_columns)


# In[ ]:


#checking the correlation b/w contact details
contact_col = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL','TARGET']
Contact_corr = application[contact_col].corr()
fig = plt.figure(figsize=(8,8))
ax = sns.heatmap(Contact_corr, xticklabels=Contact_corr.columns,yticklabels=Contact_corr.columns,annot = True,cmap ="cubehelix",linewidth=1)


# In[ ]:


#adding contact details to unwanted columns
contact_col.remove('TARGET')
unwanted_columns=unwanted_columns+contact_col
len(unwanted_columns)


# In[ ]:


unwanted_columns


# In[ ]:


#removed all unwanted columns from the application set
application.drop(labels=unwanted_columns,axis=1,inplace=True)


# In[ ]:


application.columns


# In[ ]:


application.shape


# In[ ]:


application.info(verbose=True)


# In[ ]:


#columns having more tha 40% null values converting to unwanted list
unwanted_previous=nullColumns_40prev['Column Name'].tolist()
unwanted_previous


# In[ ]:




