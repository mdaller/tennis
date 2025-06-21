#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv('tennis_stats.csv')


# In[3]:


df.head(10)
tennis = df.FirstServe, df.FirstServePointsWon, df.FirstServeReturnPointsWon, df.SecondServePointsWon, df.SecondServeReturnPointsWon, df.Aces, df.BreakPointsConverted, df.BreakPointsFaced, df.ReturnGamesWon, df.ReturnPointsWon, df.ServiceGamesPlayed, df.ServiceGamesWon, df.TotalPointsWon, df.TotalServicePointsWon, df.Wins, df.Losses, df.Winnings, df.Ranking




plt.figure(figsize=(18,12))
sns.heatmap(
    df.drop(columns=['Year']).select_dtypes(include='number').corr(),
    annot=True
)
plt.savefig("heatmap.png")
plt.show(block=True)
plt.clf()


# In[8]:




X = df[['Wins']]
y= df[['Winnings']]
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size= 0.2, random_state =50)


# In[14]:


model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)



model.score(x_test, y_test)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.title("Actual vs Predicted Winnings")
plt.savefig("actual vs predicted winings.png")
plt.show(block=True)
plt.clf()





