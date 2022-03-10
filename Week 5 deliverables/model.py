#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the dataset (Kaggle source) from our computer.
import pandas as pd
df = pd.read_csv('nba-stats-salary-rating.csv')


# In[2]:


# Dataset pre-processing step. We keep only the varibales which contain important information for our analysis.
df['Salaries'] = df['Salaries'].str.replace('$','')
df['Salaries'] = df['Salaries'].str.replace(',','')
df['Salaries'] = pd.to_numeric(df['Salaries'])
df.drop(['Unnamed: 0', 'Player', 'Tm', 'G', 'GS', 'ORB', 'DRB', 'FG', 'FGA', 'Pos', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'eFG%'], axis = 1, inplace = True)
df.rename(columns = {'MP':'Minutes_Played', 'FG%':'Fieldgoal_Percentage', '3P%':'Threepoint_Percentage', '2P%':'Twopoint_Percentage', 'FT%':'Freethrow_Percentage', 'TRB':'Total_Rebounds', 'AST':'Asists', 'STL':'Steals', 'BLK':'Blocks', 'TOV':'Turnovers', 'PF':'Personal_Fouls', 'PTS':'Points'}, inplace = True)
df['Fieldgoal_Percentage'] = df['Fieldgoal_Percentage']*100
df['Threepoint_Percentage'] = df['Threepoint_Percentage']*100
df['Twopoint_Percentage'] = df['Twopoint_Percentage']*100
df['Freethrow_Percentage'] = df['Freethrow_Percentage']*100
df


# In[3]:


# We can see the data type of each column.
df.info()


# In[4]:


# Handling missing values.
df.isnull().sum()


# In[5]:


# Fill the missing values with the mean of each column.
df['Fieldgoal_Percentage'].fillna(int(df['Fieldgoal_Percentage'].mean()), inplace = True)
df['Threepoint_Percentage'].fillna(int(df['Threepoint_Percentage'].mean()), inplace = True)
df['Twopoint_Percentage'].fillna(int(df['Twopoint_Percentage'].mean()), inplace = True)
df['Freethrow_Percentage'].fillna(int(df['Freethrow_Percentage'].mean()), inplace = True)


# In[6]:


# Indeed there are no missing values now.
df.isnull().sum()


# In[7]:


# Splitting our dataset into train and test set in order to perfrom the Machine Learning model.
from sklearn.model_selection import train_test_split
X = df.drop('Salaries', axis = 1)
y = df.iloc[:, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .8)


# In[8]:


# We utilize a simple linear regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
reg.predict(X_test)


# In[9]:


# Save the model to disk.
import pickle
pickle.dump(reg, open('model.pkl', 'wb'))

