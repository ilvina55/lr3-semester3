#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Задание 5 – Определение нестационарности


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels as sm
import sklearn as sk
pd.set_option("display.width", 100)
sns.set_style("darkgrid")
pd.plotting.register_matplotlib_converters()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import fetch_openml
bike_sharing = fetch_openml(
    "Bike_Sharing_Demand", version=2,
    as_frame=True, parser="pandas")
df = bike_sharing.frame
df.head()


# In[3]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


# In[4]:


result = adfuller(df['count'])


# In[5]:


print('ADF Test Statistic: %.2f' % result[0])
print('5%% Critical Value: %.2f' % result[4]['5%'])
print('p-value: %.2f' % result[1])


# In[6]:


result = kpss(df['count'])
print(result)
print('KPSS Test Statistic: %.2f' % result[0])
print('KPSS Test Statistic: %.2f' % result[0])
print('p-value: %.2f' % result[1])


# In[ ]:




