#import ssl
import ssl

# Bypass corporate/school network SSL certificate errors
ssl._create_default_https_context = ssl._create_unverified_context

#import seaborn, pandas and numpy

import seaborn as sns
import pandas as pd
import numpy as np

#load tips dataset
df = sns.load_dataset("tips")

print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)
print(df.info())
print(df.tail())

df[['total_bill', 'tip', 'day', 'time']].head()
df.iloc[0]
df.iloc[10:20]
df.loc[0:5, ['total_bill', 'tip']]
df.loc[20:30, ['total_bill', 'size']]

# filtering
df[df['day'] == 'Sun']
df[(df['day'] == 'Sun') & (df['total_bill'] > 20)]

# total bill > 25
print(df[df['total_bill'] > 25])
count = df[df['total_bill'] > 25].count()
print(count)

#Query
df.query("day == 'Sun' and total_bill > 20")

#Duplicate
count = df.duplicated().sum()
print(count)

#summary & Grouping
print(df.describe())
print(df.groupby('day')['tip'].mean())

# DaY with highest tip
#highest_day = average_tips.idxmax()
#highest_value = average_tips.max()

#print(average_tips)
#print(f"\nDay with the highest average tip: {highest_day} ({highest_value:.2f})")



# Tutorial 3.2
#Case Study: Titanic Passenger Survival Analysis
df = sns.load_dataset("titanic")
print(df.head())
print(df.shape)
print(df.columns)
print(df.dtypes)

# Fill missing age with median and missing embarked and embarked town with mode
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])

# Drop missing deck
df = df.drop(columns=['deck'])
print(df.columns)

#Survival rate by Sex
print(df.groupby('sex')['survived'].mean())

#Survival rate by Class
df.groupby('class')['survived'].mean()

#Survival rate by Embark town
df.groupby('embark_town')['survived'].mean()

# Filtering Females in First class
from lenses import lens
Female_first_class = df[
    (df["sex"] == "female") |
    (df["class"] == "First")
]
len(Female_first_class)

# Creating new column Age_Group
df['Age_Group'] = np.where(
    df['age'] < 18,
    'Child',
    'Adult'
)
print(df.columns)

# Survival rate

summary = (
    df.groupby(["pclass", "sex"])["survived"]
      .agg(passengers="count", survival_rate="mean")
      .reset_index()
)
summary["survival_rate"] = (summary["survival_rate"] * 100).round(1)
summary = summary.sort_values("survival_rate", ascending=False)
print(summary)