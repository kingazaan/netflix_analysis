import pandas as pd
from tabulate import tabulate
import seaborn
import matplotlib.pyplot as plt

data = pd.read_csv('Netflix_data.csv')
#print(tabulate(df, headers = 'firstrow', tablefmt = 'psql'))
print (data.head(5))
data.columns

data.corr()
plt.figure(figsize=(12,10))
seaborn.heatmap(data.corr(), annot= True, cmap = "coolwarm")


# pd.get_dummies(data.type)

#print(data.isnull().sum(axis=0))

# print("distict values:", data['rating'])
# print(data.head(5))