#!/usr/bin/env python
# coding: utf-8

# # => The Raw Data Files and their Format-

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('road-accidents.csv')


# In[3]:


df=df[9:]
df


# # => Read in and Get an Overview of the Data-

# In[4]:


df.rename(columns = {'##### LICENSE #####':'a|b|c|d|e'}, inplace = True)
split_data = df["a|b|c|d|e"].str.split("|")
data = split_data.to_list()
data


# In[5]:


names = ["a", "b","c","d","e"]
new_df = pd.DataFrame(data, columns=names)
df1=new_df.rename(columns = {'a':'State','b':'Drivers fatal collisions per billion','c':'Percentage Fatal collisions Speeding','d':'Percentage Fatal collisions Alcohol','e':'Percentage Fatal 1st time'})
df1.head(10)


# In[6]:


df1.info()


# In[7]:


df1['Drivers fatal collisions per billion']=df1['Drivers fatal collisions per billion'].astype('float64')
df1['Percentage Fatal collisions Speeding']=df1['Percentage Fatal collisions Speeding'].astype('int64')
df1['Percentage Fatal collisions Alcohol']=df1['Percentage Fatal collisions Alcohol'].astype('int64')
df1['Percentage Fatal 1st time']=df1['Percentage Fatal 1st time'].astype('int64')
df1.info()


# # => Create a Textual and a Graphical Summary of the Data-

# In[8]:


# Compute the summary statistics of all columns in the `df1` DataFrame
acc_sum = df1.describe()
acc_sum


# In[9]:


# Create a pairwise scatter plot to explore the data
g=sns.pairplot(df1)
g.fig.set_size_inches(15,15)


# # => Quantify the Association of Features and Accidents- 

# In[10]:


df1.corr()


# # => Fit a Multivariate Linear Regression-

# In[11]:


from sklearn import linear_model

# Create the DataFrames
features = df1[['Percentage Fatal collisions Speeding','Percentage Fatal collisions Alcohol','Percentage Fatal 1st time']]
target  = df1['Drivers fatal collisions per billion']

# Create a linear regression object
reg = linear_model.LinearRegression()

# Fit a multivariate linear regression model
reg.fit(features ,target)

# Retrieve the regression coefficients
fit_coef =reg.fit(features ,target).coef_
fit_coef


# # => Perform PCA on Standardized Data-

# In[12]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Import the PCA class function 
from sklearn.decomposition import PCA
pca = PCA()

# Fit the standardized data to the pca
pca.fit(features_scaled)


# In[13]:


import matplotlib.pyplot as plt
rcParams['figure.figsize'] = 10,8
plt.scatter(range(1, pca.n_components_ + 1),  pca.explained_variance_ratio_)
plt.plot(range(1, pca.n_components_ + 1),  pca.explained_variance_ratio_,color='black')
plt.xlabel('PRINCIPAL COMPONENT',fontsize=15)
plt.ylabel('PROPORTION OF VARIANCE EXPLAINED',fontsize=15)
plt.xticks([1, 2, 3])

#  The first two principal components
two_first_comp_var_exp = pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1]

print("The cumulative variance of the first two principal components is {}".format(round(two_first_comp_var_exp, 5)))


# # => Visualize the First two Principal Components-

# In[14]:


# Transform the scaled features using two principal components
pca = PCA(n_components=2)
p_comps = pca.fit_transform(features_scaled)


# In[15]:


p_comp1 = p_comps[:,0]
p_comp2 = p_comps[:,1]


# In[16]:


# Plot the first two principal components in a scatter plot
rcParams['figure.figsize'] = 10,8
plt.scatter(p_comp1,p_comp2,color = 'black')
plt.xlabel('COMPONENT_1',fontsize=15)
plt.ylabel('COMPONENT_2',fontsize=15)


# # => Find Clusters of Similar States in the Data-

# In[17]:


from sklearn.cluster import KMeans

# Explanatory power for up to 10 KMeans clusters
ks = range(1, 10)
inertias = []
for k in ks:
    # Initialize the KMeans object using the current number of clusters (k)
    km = KMeans(n_clusters=k, random_state=8)
    # Fit the scaled features to the KMeans object
    km.fit_transform(features_scaled)
    # Append the inertia for `km` to the list of inertias
    inertias.append(km.inertia_)


# In[18]:


plt.plot(list(ks), inertias, marker='o',color = 'black')
plt.xlabel('CLUSTERS',fontsize=15)
plt.ylabel('INERTIAS',fontsize=15)


# # => KMeans to Visualize Clusters in the PCA Scatter plot-

# In[19]:


km = KMeans(n_clusters=3, random_state=8)

km.fit(features_scaled)


# In[20]:


plt.scatter(p_comps[:, 0], p_comps[:, 1],label=km, c=km.labels_)
plt.xlabel('PRINCIPLE COMPONENT_1',fontsize=15)
plt.ylabel('PRINCIPLE COMPONENT_2',fontsize=15)


# # => Visualize the Feature Differences between the Clusters-

# In[21]:


# Create a new column with the labels from the KMeans clustering
df1['cluster'] =km.labels_

# Reshape the DataFrame to the long format
clus_car = pd.melt(df1 ,id_vars='cluster' ,var_name='measurement', value_name='percent',value_vars=['Percentage Fatal collisions Speeding','Percentage Fatal collisions Alcohol','Percentage Fatal 1st time'])
clus_car.info()


# In[22]:


sns.violinplot(x=clus_car['percent'], y=clus_car['measurement'])
plt.xlabel('PERCENT',fontsize=15)
plt.ylabel('MEASUREMENT',fontsize=15)


# In[23]:


df2=clus_car.groupby(['cluster','measurement']).mean().reset_index()
df2.info()


# In[24]:


sns.violinplot(x=df2['percent'], y=df2['measurement'])
plt.xlabel('PERCENT',fontsize=15)
plt.ylabel('MEASUREMENT',fontsize=15)


# # => Compute the Number of Accidents with in each Cluster-

# In[25]:


df_1=pd.read_csv('miles-driven.csv')
df_1.columns


# In[26]:


split_data = df_1["state|million_miles_annually"].str.split("|")
df_1= split_data.to_list()
df_1


# In[27]:


names=['a','b']
df_2 = pd.DataFrame(df_1, columns=names)
df_2=df_2.rename(columns = {'a':'State','b':'Million Miles Annually'})
df_2.head(10)


# In[28]:


# Merge the `df1` DataFrame with the `df_2` DataFrame
Car_Acc = pd.merge(df1, df_2, on='State')
Car_Acc


# In[29]:


Car_Acc.info()


# In[30]:


Car_Acc['Million Miles Annually']=Car_Acc['Million Miles Annually'].astype('float64')
Car_Acc.info()


# In[31]:


# Create a new column for the number of drivers involved in fatal accidents
Car_Acc['Number of Drivers fatal collisions']=(Car_Acc['Drivers fatal collisions per billion']/1000)*Car_Acc['Million Miles Annually']


# In[32]:


Car_Acc.head(10)


# In[33]:


Car_Acc.describe()


# In[34]:


sns.barplot(x=Car_Acc['cluster'], y=Car_Acc['Number of Drivers fatal collisions'], data=Car_Acc, estimator=sum,color='darkgray', ci=None)
plt.ylabel('NUMBER OF DRIVERS FATAL COLLISIONS',fontsize=15)
plt.xlabel('CLUSTER',fontsize=15)


# In[35]:


Mean_count = Car_Acc[['cluster','Number of Drivers fatal collisions']]
Mean_count.groupby('cluster').agg(['count','mean','sum'])


# # => Make a Decision When there is no Clear Right Choice-

# In[36]:


# Which cluster would you choose?
cluster_num = 2

