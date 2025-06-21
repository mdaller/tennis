#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv('tennis_stats.csv')


# In[3]:


df.head(10)
tennis = df.FirstServe, df.FirstServePointsWon, df.FirstServeReturnPointsWon, df.SecondServePointsWon, df.SecondServeReturnPointsWon, df.Aces, df.BreakPointsConverted, df.BreakPointsFaced, df.ReturnGamesWon, df.ReturnPointsWon, df.ServiceGamesPlayed, df.ServiceGamesWon, df.TotalPointsWon, df.TotalServicePointsWon, df.Wins, df.Losses, df.Winnings, df.Ranking


# In[4]:


#sns.heatmap(tennis)


# In[5]:


sns.scatterplot(x=df.Winnings, y= df.Wins, hue = df.Ranking, data = df)


# In[6]:


df.head()


# In[7]:


#heat = df.Wins, df.Losses, df.Aces
#sns.heatmap(heat)


# In[8]:


columns =df.columns
compare = [col for col in df.columns if col  != 'Player']


# In[9]:


# winngs vs aces left skewed data
'return points won, services games won right skewed'
# wins, losses break points faced, break point oportunties, return games played, servicesgp postive correlation
for column in compare:
    plt.figure()
    sns.scatterplot(x= column , y= 'Winnings', data = df)
    #plt.xlabel("winnings ")
#plt.show()


# In[11]:


X = df[['Wins']]
y= df[['Winnings']]
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size= 0.2, random_state =50)


# In[14]:


model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)





# In[13]:


model.score(x_test, y_test)


# In[ ]:





# In[ ]:




