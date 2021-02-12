# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import matplotlib.pyplot as plot
from matplotlib.pyplot import figure
import plotly.graph_objects as go
import seaborn as sb
import numpy as np
import pandas as pd
import sklearn as sk


# %%
data = pd.read_csv('Netflix_data.csv')
#print(tabulate(df, headers = 'firstrow', tablefmt = 'psql'))
print(data.head(5))
data.columns


# %%
data.nunique()


# %%
data["date_added"] = pd.to_datetime(data['date_added'])
data['day_added'] = data['date_added'].dt.day
data['year_added'] = data['date_added'].dt.year
data['month_added']= data['date_added'].dt.month
data['year_added'].astype(float)
data['day_added'].astype(float)
data.head()


# %%
figure(figsize = (30, 15))


# %%
plot.pie(data.type.value_counts(), labels = data.type.unique(), autopct='%1.1f%%', pctdistance = .5, startangle = 70)


# %%
data.dropna(subset = ['date_added', 'rating'], how = 'any').shape
data = data.dropna(subset = ['date_added', 'rating'], how = 'any')


# %%
fig = plot.subplots(figsize = (16,8))
plot.pie(data.rating.value_counts(), labels = data.rating.unique(), autopct='%1.1f%%', pctdistance = .7, startangle = 70)


# %%
d = data.groupby(['type', 'rating'])[['type']].count()
d


# %%
movie_ratings = d.drop(index = 'TV Show')
tv_ratings = d.drop(index = 'Movie')


# %%
print(movie_ratings)
print(tv_ratings)


# %%
movie_ratings = movie_ratings.droplevel(level = 0)
tv_ratings = tv_ratings.droplevel(level = 0)


# %%
movie_ratings = movie_ratings.reset_index()
tv_ratings = tv_ratings.reset_index()


# %%
print(movie_ratings)
print(tv_ratings)


# %%
col = 'rating'

vc1 = movie_ratings[col].value_counts().reset_index()
print(vc1)
vc1 = vc1.rename(columns = {col : "count", "index" : col})
vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))
vc1 = vc1.sort_values(col)

vc2 = tv_ratings[col].value_counts().reset_index()
vc2 = vc2.rename(columns = {col : "count", "index" : col})
vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))
vc2 = vc2.sort_values(col)

trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))
trace2 = go.Bar(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))
data = [trace1, trace2]
layout = go.Layout(title="Content by rating", legend=dict(x=0.1, y=1.1, orientation="h"))
fig = go.Figure(data, layout=layout)
fig.show()


# %%



