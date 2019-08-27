#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import statistics 
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import plot_importance
from xgboost import XGBRegressor as xgb
import matplotlib.pyplot as plt
from IPython.display import HTML, display, clear_output
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(18,6)})
import os
# os.listdir()


# In[2]:


# import train and test dataset
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')


# In[3]:


print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


# In[4]:


# print columns in training dataset
train.columns


# In[5]:


train.describe()


# In[6]:


# plot training dataset heatmap
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# In[7]:


# drop columns array
drop_col_array = [  'loan_id' ]
print(drop_col_array)


# In[8]:


# drop loan_id
train = train.drop(drop_col_array, axis=1)
test = test.drop(drop_col_array, axis=1)
print(train.shape, test.shape)


# In[9]:


# checking missing data percentage in train data
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing_train  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_train.head(30)


# In[10]:


# print data types of each column
train.dtypes


# In[11]:


# Print number of unique elements in each column
for column in train.columns:
    print(train[column].nunique(),"  ", column)


# # Data Visualization

# In[12]:


train.head()


# In[13]:


# This function returns the count plot of a column with percentage of each class
def plot_bar_counts_categorical(data_se, title, figsize, sort_by_counts=False):
    info = data_se.value_counts()
    info_norm = data_se.value_counts(normalize=True)
    categories = info.index.values
    counts = info.values
    counts_norm = info_norm.values
    fig, ax = plt.subplots(figsize=figsize)
    if data_se.dtype in ['object']:
        if sort_by_counts == False:
            inds = categories.argsort()
            counts = counts[inds]
            counts_norm = counts_norm[inds]
            categories = categories[inds]
        ax = sns.barplot(counts, categories, orient = "h", ax=ax)
        ax.set(xlabel="count", ylabel=data_se.name)
        ax.set_title("Distribution of " + title)
        for n, da in enumerate(counts):
            ax.text(da, n, str(da)+ ",  " + str(round(counts_norm[n]*100,2)) + " %", fontsize=10, va='center')
    else:
        inds = categories.argsort()
        counts_sorted = counts[inds]
        counts_norm_sorted = counts_norm[inds]
        ax = sns.barplot(categories, counts, orient = "v", ax=ax)
        ax.set(xlabel=data_se.name, ylabel='count')
        ax.set_title("Distribution of " + title)
        for n, da in enumerate(counts_sorted):
            ax.text(n, da, str(da)+ ",  " + str(round(counts_norm_sorted[n]*100,2)) + " %", fontsize=10, ha='center')


# In[14]:


plot_bar_counts_categorical(train['source'], 'Train dataset: source', (5,5))


# In[15]:


plot_bar_counts_categorical(train['financial_institution'], 'Train dataset: financial_institution', (5,5))


# In[16]:


plt.figure(figsize=(20, 2))
plt.plot(train['interest_rate'][:500])
plt.title('location_y')
plt.show()


# In[17]:


plot_bar_counts_categorical(train['origination_date'], 'Train dataset: origination_date', (5,5))


# In[18]:


plt.figure(figsize=(20, 2))
plt.plot(train['loan_to_value'][:1000])
plt.title('location_y')
plt.show()


# In[19]:


plt.figure(figsize=(20, 2))
plt.plot(train['number_of_borrowers'][:1000])
plt.title('location_y')
plt.show()


# In[20]:


plt.figure(figsize=(20, 2))
plt.plot(train['debt_to_income_ratio'][:1000])
plt.title('location_y')
plt.show()


# In[21]:


plot_bar_counts_categorical(train['insurance_type'], 'Train dataset: insurance_type', (5,5))


# In[22]:


plt.figure(figsize=(20, 2))
plt.plot(train['m1'][:])
plt.title('location_y')
plt.show()


# In[23]:


plt.figure(figsize=(20, 2))
plt.plot(train['m13'][:])
plt.title('location_y')
plt.show()


# In[24]:


plot_bar_counts_categorical(train['m13'], 'Train dataset: m13', (5,5))


# In[25]:


plt.figure(figsize=(7, 3))
sns.boxplot(train["m1"])
plt.show()


# In[26]:


plt.figure(figsize=(7, 3))
sns.boxplot(train["m2"])
plt.show()


# In[27]:


plt.figure(figsize=(7, 3))
sns.boxplot(train["m5"])
plt.show()


# In[28]:


plt.figure(figsize=(7, 3))
sns.boxplot(train["m9"])
plt.show()


# In[ ]:





# In[29]:


plt.figure(figsize=(7, 3))
sns.boxplot(train["borrower_credit_score"])
plt.show()


# In[30]:


plt.figure(figsize=(7, 3))
sns.boxplot(train["loan_to_value"])
plt.show()


# In[31]:


plt.figure(figsize=(7, 3))
sns.boxplot(train["interest_rate"])
plt.show()


# In[ ]:





# In[ ]:





# Label Encoding

# In[32]:


# Label Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

for x in train.columns:
    if train[x].dtype == type(object):
        train[x] = train[x].fillna('NaN')
        test[x] = test[x].fillna('NaN')
        encoder = LabelEncoder()
        encoder.fit(list(set(list(train[x]) + list(test[x]))))
        train[x] = encoder.transform(train[x])
        test[x] = encoder.transform(test[x])


# In[33]:


train.head()


# In[34]:


test.head()


# In[35]:


print(train.shape)
print(test.shape)


# # Model Training

# In[36]:


# Splitting training dataset into train and test
X = train.copy().drop('m13', axis=1).values
y = train['m13']


# In[37]:


print(train.shape)
print(X.shape)
print(y.shape)


# In[38]:


X[:2]


# In[39]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_whole = sc.transform(X.copy())
test_v = sc.transform(test.copy().values)


# In[40]:


X[0], X_train[0]


# In[41]:


test_v


# # XGBoost

# In[42]:


# XGB Classifier
from xgboost import XGBClassifier

classifier = XGBClassifier( learning_rate =0.1,
 n_estimators=112,
 max_depth=9,
 min_child_weight=5,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.6,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=13,
 reg_lambda=5,
# max_delta_step=1,
 alpha=0,
 base_score=0.5,
 seed=1029)

classifier.fit(X_train, y_train)


# In[43]:


# plot feature importance
plot_importance(classifier)
plt.figure(figsize=(200, 200))
plt.show()


# In[44]:


print(classifier.feature_importances_)


# In[45]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred


# In[46]:


# print f1 score
from sklearn.metrics import f1_score
f1_score(y_test, y_pred)


# In[47]:


test_pred = classifier.predict(test_v)


# In[48]:


print(test_pred.shape)
test_pred[:10]


# In[49]:


# print number of 1s and 0s in predicted values

unique, counts = np.unique(test_pred, return_counts=True)
dict(zip(unique, counts))


# In[50]:


# load loan_id of test dataset
test_loan_id = pd.read_csv('dataset/test.csv')['loan_id']
print(test_loan_id.shape)


# In[51]:


# save results to csv
subm = pd.DataFrame({'loan_id': test_loan_id, 'm13': test_pred})
subm = subm[['loan_id','m13']]    

filename='solution/AakashJhawar_16011999_final.csv'
subm.to_csv(filename, index=False)

