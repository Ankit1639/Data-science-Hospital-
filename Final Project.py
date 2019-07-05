
# coding: utf-8

# # Project by Team: Data Solutions
#     Members: Ankit Yadav 
#              Kumar Roushan
#               Simran Kashayap
#                 Nitish
#         ----------------------------------------------------------

# # About:
#       Here we are having Dataset of Hospitals of a Country.
#       We having out pateint payment Records. Here we will 
#         work over the total no of out pateints and many monre.

# @ Import the Csv File:

# In[1]:


import pandas as pd
data=pd.read_csv("C:\\Users\\User\\Desktop\\hospital.csv")


# @Displaying the imported file

# In[2]:


data


# @Displaying only Five Records from file

# In[3]:


data.head()


# @taking values from Csv within the variable

# In[4]:


X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


# @displaying value in Variable

# In[5]:


X


# @displaying value in Variable

# In[6]:


y


# @Taking care of Missing Data

# In[7]:


print (data.isnull())


# @to check Missing Data in particular Record

# In[8]:


print (data['Provider Id'].isnull())


# @To check total No of Errors or Missing Data in the Csv

# In[9]:


print (data.isnull().sum())


# @to check the value missing in the csv 

# In[10]:


print ( data.isnull().values.any())


# In[11]:


print (data.isnull().sum().sum())


# In[12]:


data.describe()


# In[13]:


data.describe(include=['O'])


# @For group by Operator

# In[14]:


for grp, dat in data.groupby("Provider Id"):
    print(grp, data)


# # Data Visualization Pandas

# In[15]:


import numpy as np 


# @Graph using Average Total Payment

# In[16]:


data['Average Total Payments'].plot(kind='hist',bins=50,figsize=(8,6))


# @Graph using Average Total Payment and Toatl no of Outpatient Services

# In[17]:


data.plot(kind='hist',x='Outpatient Services',y='Average Total Payments')


# @to display all values from file

# In[18]:


data.plot(kind='hist',figsize=(8,6))


# @scatter graph between Outpatient Services and Average Total Payments

# In[19]:


data.plot(kind='scatter',x='Outpatient Services',y='Average Total Payments',figsize=(8,6))


# @Kernel Density Estimation plot (KDE) for Average Total Payments

# In[20]:


data['Average Total Payments'].plot(kind='kde')


# # Regression:

# In[24]:


data.head()


# In[27]:


h=data['Average Total Payments'].values
h.reshape(-1,1)
h=pd.DataFrame(h)


# @reshaping the values

# In[28]:


h


# In[30]:


i=data['Average Estimated Submitted Charges'].values
i.reshape(-1,1)
i=pd.DataFrame(i)


# In[31]:


i


# @Standardising the file using the Sklearn librabry

# In[33]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scale = StandardScaler()


# @Reverifying the variable

# In[34]:


scale


# @Training and testing

# In[36]:


from sklearn.model_selection import train_test_split


# In[38]:


h_train, h_test, i_train,i_test =train_test_split(h,i,test_size=0.3,random_state=2)


# @displaying the values in the variables

# In[40]:


h_train


# In[41]:


h_test


# In[42]:


i_train


# In[43]:


i_test


# # @Linear Regression

# In[46]:


from sklearn.linear_model import LinearRegression


# In[47]:


reg=LinearRegression()


# @using the pipeline concept

# In[48]:


pipe=make_pipeline(scale,reg)


# In[49]:


pipe


# In[51]:


pipe.fit(h_train,i_train)


# @using the Predict function

# In[52]:


reg_pre=pipe.predict(h_test)


# In[53]:


reg_pre


# @predict the Accuracy using score function

# In[54]:


pipe.score(h_test,i_test)


# In[55]:


data.corr()


# @displaying the corelation between data in the csv file

# In[56]:


from statsmodels.api import graphics as sm
sm.plot_corr(data.corr(), xnames=list(data.columns))


# @graph using matplotlib.pyplot

# In[57]:


import matplotlib.pyplot as plt


# In[58]:


plt.plot(h,i)


# In[59]:


plt.scatter(h,i)


# # Lasso Regression

# @Using the Lasso Regression for upgradation and Accuracy Checking

# In[60]:


from sklearn.linear_model import Lasso
la=Lasso()


# In[61]:


p=make_pipeline(scale,la)


# In[62]:


p.fit(h_train,i_train)


# In[63]:


p.score(h_test,i_test)


# In[ ]:


#Project Finished with Accuracy Rate By Linear Regression:-0.05772830330639955
                                    #by Lasso Regression:-0.055585752528895194

