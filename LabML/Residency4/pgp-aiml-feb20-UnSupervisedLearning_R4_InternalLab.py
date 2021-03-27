#!/usr/bin/env python
# coding: utf-8

# # Principal Component Analysis with Cancer Data

# In[1]:


#Import all the necessary modules
#Import all the necessary modules
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Q1. Load the Data file ( Breast Cancer CSV) into Python DataFrame and view top 10 rows

# In[2]:


dfCancer = pd.read_csv("/home/amol/GL/LabML/Residency4/Cancer.csv")


# In[3]:


dfCancer.head(10)


# In[ ]:


# Id columns is to identify rows hence can be skipped in analysis
# All columns have numerical values
# Class would be the target variable. Should be removed when PCA is done


# # Q2 Print the datatypes of each column and the shape of the dataset. Perform descriptive analysis 

# In[4]:


dfCancer.dtypes


# In[5]:


dfCancer.shape


# In[6]:


dfCancer.info()


# In[7]:


dfCancer.describe().transpose()


# # Q3 Check for missing value check, incorrect data, duplicate data and perform imputation with mean, median, mode as necessary.
# 

# In[ ]:


# We could see "?" values in column, this should be removed from data set

# Check for missing value in any other column


# In[14]:


#Function to check if any column has null values
def CheckifNullExists(sr) :
    if (sr.isnull().sum() >0) :
        return ("Column" + " " + "'" + sr.name + "'" + " " + "has " + sr.isnull().sum() + " values")
    else :
        return ("Column" + " " + "'" + sr.name + "'" + " " + "has no null values")


# In[15]:


# Check if any column has null values
for (columnName, columnData) in dfCancer.iteritems():
    print(CheckifNullExists (columnData))


# In[13]:


dfCancer['Bare Nuclei'].unique()


# In[17]:


dfCancer.isna().sum()


# In[ ]:


# No missing values found. So let us try to remove ? from bare nuclei column

# Get count of rows having ?


# In[19]:


dfCancer['Bare Nuclei'].value_counts()


# In[ ]:


# 16 values are corrupted. We can either delete them as it forms roughly 2% of data.
# Here we would like to impute it with suitable values


# In[20]:


#Lets repalce ? with 1 as number of 1 is 402
dfCancer['Bare Nuclei'].replace("?","1",inplace=True)
dfCancer['Bare Nuclei'].value_counts()


# In[23]:


#Lets check for duplicate records
duplicate = dfCancer.duplicated()
print(duplicate.sum())


# In[25]:


dfCancer[duplicate]


# In[34]:


dfCancer.drop_duplicates(subset=None, keep="first", inplace=True)


# In[35]:


dfCancer.shape


# In[36]:


dul = dfCancer.duplicated()
print(dul.sum())


# In[38]:


#Change data type of 'Bare Nuclei' to int64
dfCancer['Bare Nuclei'] = dfCancer['Bare Nuclei'].astype('int64')


# In[39]:


dfCancer.dtypes


# # Q4. Perform bi variate analysis including correlation, pairplots and state the inferences. 
# 

# In[ ]:


# Check for correlation of variable


# In[40]:


dfCancer.corr()


# In[ ]:


# Cell size shows high significance with cell shape,marginal adhesion, single epithelial cell size,bare nuclei, normal nucleoli 
# and bland chromatin
# Target variable shows high correlation with most of these variables


# Inferences based on corelation 
# 1) Cell size and Cell Shape - shows very high co-relation. 
# 2) Cell size and Normal Nucleoli - show high co-relation
# 3) Cell size shows high co-relation with almost all other fields except 'Mitoses' field

# In[42]:


#Lets take back up of data set
dfCancer_backup = dfCancer


# In[47]:


#drop Id column 
dfCancer= dfCancer.drop(['ID'],axis=1)


# In[ ]:


#Let us check for pair plots


# In[49]:


sns.pairplot(dfCancer, diag_kind='kde', hue='Class')


# In[ ]:


# Relationship between variables shows come correlation. 
# Distribution of variables shows most of the values are concentrated on lower side, though range remains same for all that is 
# Between 1 to 10


# Inferences based on corelation 
# 1) ClumThickness - this field shows good amount classification
# 2) The distribution of Cell size, cell shape, Marginal Adhesion, Single Epithelial Cell Size for class 4 - do not see any peaks...most the distribution is bulgy and have low values

# # Q5 Remove any unwanted columns or outliers, standardize variables in pre-processing step

# In[ ]:


# We could see most of the outliers are now removed.


# In[50]:


dfCancer.boxplot(figsize=(20,3))


# Inferences based on corelation 
# 1) We can see outliers in fields - Marginal Adhesion Signle, Single Epithelial Cell size, Normal Nucleoli, Brand Chromatin, mitoses

# In[51]:


cross = pd.crosstab(dfCancer['Marginal Adhesion'],dfCancer['Class'])
cross


# In[52]:


dfCancer['Marginal Adhesion'] = np.where(dfCancer['Marginal Adhesion'] > 6,6,dfCancer['Marginal Adhesion'])


# In[54]:


dfCancer.boxplot(figsize=(20,3))


# In[53]:


cross = pd.crosstab(dfCancer['Single Epithelial Cell Size'],dfCancer['Class'])
cross


# In[55]:


dfCancer['Single Epithelial Cell Size'] = np.where(dfCancer['Single Epithelial Cell Size'] > 6,6,dfCancer['Single Epithelial Cell Size'])


# In[56]:


dfCancer.boxplot(figsize=(20,3))


# In[57]:


cross = pd.crosstab(dfCancer['Normal Nucleoli'],dfCancer['Class'])
cross


# In[58]:


dfCancer['Normal Nucleoli'] = np.where(dfCancer['Normal Nucleoli'] > 6,6,dfCancer['Normal Nucleoli'])


# In[59]:


dfCancer.boxplot(figsize=(20,3))


# In[60]:


cross = pd.crosstab(dfCancer['Bland Chromatin'],dfCancer['Class'])
cross


# In[61]:


dfCancer['Bland Chromatin'] = np.where(dfCancer['Bland Chromatin'] > 5,5,dfCancer['Bland Chromatin'])


# In[62]:


dfCancer.boxplot(figsize=(20,3))


# In[63]:


cross = pd.crosstab(dfCancer['Mitoses'],dfCancer['Class'])
cross


# In[64]:


dfCancer['Mitoses'] = np.where(dfCancer['Mitoses'] > 3,3,dfCancer['Mitoses'])


# In[65]:


dfCancer.boxplot(figsize=(20,3))


# In[66]:


#Lets drop class column
dfCancer_new = dfCancer.drop('Class',axis=1)


# In[69]:


dfCancer_new.shape


# In[71]:


from scipy.stats import zscore


# In[72]:


dfCancer_new_std = dfCancer_new.apply(zscore)


# In[74]:


dfCancer_new_std.head(10)


# # Q6 Create a covariance matrix for identifying Principal components

# In[77]:


# PCA
# Step 1 - Create covariance matrix
cov_matrix = np.cov(dfCancer_new_std.T)
print(cov_matrix)


# In[ ]:





# # Q7 Identify eigen values and eigen vector

# In[ ]:


# Step 2- Get eigen values and eigen vector


# In[78]:


eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)


# In[79]:


print("Eigen Values", eigen_values)
print("Eigen Vectors", eigen_vectors)


# # Q8 Find variance and cumulative variance by each eigen vector

# In[81]:


total = sum(eigen_values)
var_exp = [(i/total)*100 for i in sorted(eigen_values,reverse=True)]


# In[82]:


plt.plot(var_exp)


# In[83]:


cum_var_exp = np.cumsum(var_exp)
print ("Cumulative Variance Explained", cum_var_exp)


# In[92]:


plt.bar(range(1,eigen_values.size+1),var_exp,align='center')
plt.step(range(1,eigen_values.size+1),cum_var_exp, where='mid')
plt.ylabel('Explained Variance')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.show()


# # Q9 Use PCA command from sklearn and find Principal Components. Transform data to components formed

# In[93]:


from sklearn.decomposition import PCA


# In[94]:


#Lets consider 7 components as more than 95 % variance is captured by 7 features
pca = PCA(n_components=7)


# In[95]:


reduced_dataset = pca.fit_transform(dfCancer_new_std)


# In[96]:


reduced_dataset


# In[97]:


pca.components_


# # Q10 Find correlation between components and features

# In[99]:


list(dfCancer_new_std)


# In[100]:


df_comp_feature  = pd.DataFrame(pca.components_,columns=list(dfCancer_new_std))


# In[101]:


df_comp_feature


# In[102]:


sns.heatmap(df_comp_feature)


# In[ ]:




