#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plot
from matplotlib.pyplot import figure
import plotly.graph_objects as go
import seaborn as sb
import numpy as np
import pandas as pd
import sklearn as sk


data = pd.read_csv('Netflix_data.csv')
#print(tabulate(df, headers = 'firstrow', tablefmt = 'psql'))
print(data.head(5))
data.columns



data.nunique()



data["date_added"] = pd.to_datetime(data['date_added'])
data['day_added'] = data['date_added'].dt.day
data['year_added'] = data['date_added'].dt.year
data['month_added']= data['date_added'].dt.month
data['year_added'].astype(float)
data['day_added'].astype(float)
data.head()


figure(figsize = (30, 15))


plot.title('Percentage of Movies to TV Shows in Dataset')
plot.pie(data.type.value_counts(), labels = data.type.unique(), autopct='%1.1f%%', pctdistance = .5, startangle = 70)
plot.show()



data.dropna(subset = ['date_added', 'rating'], how = 'any').shape
data = data.dropna(subset = ['date_added', 'rating'], how = 'any')


fig = plot.subplots(figsize = (16,8))
plot.title('Percentage of Movies to TV Shows in Dataset')
plot.pie(data.rating.value_counts(), labels = data.rating.unique(), autopct='%1.1f%%', pctdistance = .7, startangle = 70)
plot.show()


d = data.groupby(['type', 'rating'])[['type']].count()


movie_ratings = d.drop(index = 'TV Show')
tv_ratings = d.drop(index = 'Movie')


print(movie_ratings)
print(tv_ratings)


movie_ratings = movie_ratings.droplevel(level = 0)
tv_ratings = tv_ratings.droplevel(level = 0)


movie_ratings = movie_ratings.reset_index()
tv_ratings = tv_ratings.reset_index()


print(movie_ratings)
print(tv_ratings)


col = 'rating'

vc1 = movie_ratings[col].value_counts().reset_index()
vc1['rating'] = movie_ratings['type']
vc1 = vc1.rename(columns = {col : "count", "index" : col})
vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))
vc1 = vc1.sort_values(col)

vc2 = tv_ratings[col].value_counts().reset_index()
vc2['rating'] = tv_ratings['type']
vc2 = vc2.rename(columns = {col : "count", "index" : col})
vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))
vc2 = vc2.sort_values(col)

trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="Movies", marker=dict(color="#a678de"))
trace2 = go.Bar(x=vc2[col], y=vc2["count"], name="TV Show", marker=dict(color="#6ad49b"))
dataa = [trace1, trace2]
layout = go.Layout(title="Content by rating", legend=dict(x=0.1, y=1.1, orientation="h"))
fig = go.Figure(dataa, layout=layout)
fig.show()


p = data[['type', 'title', 'duration']]
p['title_length'] = p['title'].str.len()
p = p.drop('title', 1)


name_split = p['duration'].str.split()
p['int_duration'] = name_split.str[0]
p['int_duration'] = pd.to_numeric(p['int_duration'])
p.dtypes
p = p.drop('duration', 1)


movie_t_id = p[p['type'] == 'Movie']
TV_t_id = p[p['type'] == 'TV Show']


movie_t_id = movie_t_id.rename(columns = {'int_duration': 'movie length'})
TV_t_id = TV_t_id.rename(columns = {'int_duration': 'number of seasons'})


TV_t_id
sb.scatterplot(x = 'movie length', y = 'title_length', data = movie_t_id)
sb.scatterplot(x = 'number of seasons', y = 'title_length', data = TV_t_id)


q = data[['type', 'country', 'duration']]
name_split = q['duration'].str.split()
q['int_duration'] = name_split.str[0]
q['int_duration'] = pd.to_numeric(q['int_duration'])
q.dtypes
q = q.drop('duration', 1)


movie_country = q[q['type'] == 'Movie']
TV_country = q[q['type'] == 'TV Show']
movie_country = movie_country.rename(columns = {'int_duration': 'movie length'})
TV_country = TV_country.rename(columns = {'int_duration': 'number of seasons'})


TV_country['number of seasons'].value_counts()