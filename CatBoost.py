#!/usr/bin/env python
# coding: utf-8

# # can use snpashot_file and snapshot_interval so that we can run a model that takes a long time, and continue where we left off


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn
import catboost
from pylab import rcParams



data = pd.read_csv('Netflix_data.csv')
#print(tabulate(df, headers = 'firstrow', tablefmt = 'psql'))
print(data.head(5))
data.columns



data.isnull().sum(axis=0)
data = data.drop(['director'], axis = 1)
data.head(5)
data.isnull().sum(axis=0)/len(data)*100
data.shape
data.dropna(subset = ['date_added', 'rating'], how = 'any').shape
data = data.dropna(subset = ['date_added', 'rating'], how = 'any')
data.dropna(subset = ['country', 'cast'], how = 'all').shape
data = data.dropna(subset = ['country', 'cast'], how = 'all')


  


data['country'].value_counts(dropna = False)
data[data['country'].isnull()]
data.isnull().sum(axis=0)
data['country'] = data['country'].fillna('Other')
data['cast'] = data['cast'].fillna('Unknown')
data.isnull().sum(axis = 0)


  


# get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')


# ## I used all variables as chi square candidate below

# # data[data.columns[0]]
# from scipy.stats import chi2_contingency
# for i in range(11):
#     n = 'description'
#     table = pd.crosstab(data[n], data[data.columns[i]])
#     chi2, p, dof, expected = chi2_contingency(table.values)
#     print('Chi square result for columns %s and: %s' % (n, data.columns[i]))
#     print('Chi-stat %0.3f p_value %0.3f' % (chi2, p))

# data["date_added"] = pd.to_datetime(data['date_added'])
# data['day_added'] = data['date_added'].dt.day
# data['year_added'] = data['date_added'].dt.year
# data['month_added']= data['date_added'].dt.month
# data['year_added'].astype(string)
# data['day_added'].astype(string)
# data.head()

# data.corr()
# plt.figure(figsize=(12,10))
# seaborn.heatmap(data.corr(), annot= True, cmap = "coolwarm")

data.nunique()


  


from sklearn.model_selection import train_test_split
x = data.drop(['show_id', 'type'], axis = 1)
y = data['type']
cat_features = list(range(0, x.shape[1]))
print(cat_features)


# ## I used this in order to view how many of each (TV or MOvie) were in each column, and i encoded each with numbers so that it would be summable

  


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
temp = le.fit_transform(y)
print('Labels: {}'.format(set(temp)))
print('Zero count = {}, One count = {}'.format(len(temp) - sum(temp), sum(temp)))
le.classes_


  


from catboost import Pool
pool = Pool(data = x, label = y, cat_features = cat_features)
data = train_test_split(x, y, test_size = 0.2, random_state = 176)
x_train, x_validation, y_train, y_validation = data

train_pool = Pool(
    data = x_train,
    label = y_train,
    cat_features = cat_features)

validation_pool = Pool(
    data = x_validation,
    label = y_validation,
    cat_features = cat_features)


from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations = 1000,
    verbose = 100,
    loss_function = 'Logloss',
    custom_loss = ['AUC', 'Accuracy'])


# # Can't use the "plot" function in the model because catboost plots are it is incompatible with VS code

model.fit(
    train_pool,
    plot = True,
    eval_set = validation_pool,
    verbose = 100
    )
#plot = True


  


print('Model is fitted: {}'.format(model.is_fitted()))
print('Model params\n{}'.format(model.get_params()))


# ## Not really working, believe it is because it takes too much cpu

# #used for cross-validation, to find best learning rate
# from catboost import cv
# 
# params = {
#     'loss_function': 'Logloss',
#     'iterations': 100,
#     'learning_rate': 0.5}
# 
# cv_data = cv(
#     params = params,
#     pool = train_pool,
#     fold_count = 5,
#     shuffle = True,
#     partition_random_seed = 0,
#     verbose = False)

# from sklearn.model_selection import train_test_split
# x = data.drop(['show_id', 'type'], axis = 1)
# y = data['type']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 176)

  


model = CatBoostClassifier(iterations = 1000, learning_rate = 0.04694)
model.fit(train_pool, verbose = 100)


# # 3 ways to predict with this model

  


print(model.predict(x_validation))


  


print(model.predict_proba(x_validation))


  


raw_pred = model.predict(
    x_validation,
    prediction_type = 'RawFormulaVal')

print(raw_pred)


  


from numpy import exp

sigmoid = lambda x: 1 / (1 + exp(-x))
probabilities = sigmoid(raw_pred)
print(probabilities)


  


import matplotlib.pyplot as pyplot
from catboost.utils import get_roc_curve
from catboost.utils import get_fpr_curve
from catboost.utils import get_fnr_curve

curve = get_roc_curve(model, validation_pool)
(fpr, tpr, thresholds) = curve
(thresholds, fpr) = get_fpr_curve(curve = curve)
(thresholds, fnr) = get_fnr_curve(curve = curve)


  


plt.figure(figsize = (16, 8))
style = {'alpha': 0.5, 'lw': 12}

plt.plot(thresholds, fpr, color = 'blue', label = 'FPR', **style)
plt.plot(thresholds, fnr, color = 'green', label = 'FNR', **style)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.grid(True)
plt.xlabel('Threshold', fontsize = 16)
plt.ylabel('Error Rate', fontsize = 16)
plt.title('FPR-FNR curves', fontsize = 20)
plt.legend(loc = 'lower left', fontsize = 16)


  


np.array(model.get_feature_importance(prettified = True))


  


np.array(model.get_feature_importance(
    train_pool,
    type = 'LossFunctionChange',
    prettified = True
))


  


print(model.predict_proba([x.iloc[1, :]]))
print(model.predict_proba([x.iloc[91, :]]))


  


shap_values = model.get_feature_importance(
    validation_pool,
    type = 'ShapValues')

expected_value = shap_values[0,-1]
shap_values = shap_values[:, :-1]
print(shap_values.shape)


  


proba = model.predict_proba([x.iloc[1, :]])[0]
raw = model.predict([x.iloc[1, :]], prediction_type = 'RawFormulaVal')[0]
print('Probabilities', proba)
print('Raw formula value %.4f' % raw)
print('Probability from raw value %.4f' % sigmoid(raw))


  


import shap
shap.initjs()
shap.force_plot(expected_value, shap_values[1, :], x_validation.iloc[1, :])


  


proba = model.predict_proba([x.iloc[-1, :]])[0]
raw = model.predict([x.iloc[-1, :]], prediction_type = 'RawFormulaVal')[0]
print('Probabilities', proba)
print('Raw formula value %.4f' % raw)
print('Probability from raw value %.4f' % sigmoid(raw))


  


shap.initjs()
shap.force_plot(expected_value, shap_values[-1, :], x_validation.iloc[-1, :])


  


shap.summary_plot(shap_values, x_validation)


# ## couldnt figure out the confustion matrix

  


from catboost.utils import get_confusion_matrix
conf_matrix = get_confusion_matrix(model, Pool(x_train, y_train))
print(conf_matrix)


  


features_display = x.loc[cat_features]
shap.summary_plot(shap_values, features_display, plot_type = 'bar')


# ## couldn't get to work because of error:
# EOL while scanning string literal

  


shap.dependence_plot(ind = 'duration',  interaction_index = 'listed_in', shap_values = shap_values, features = x_validation[:-1], display_features = features_display)


# ## Doesn't worklk for a complicated reason: something like this -> https://github.com/slundberg/shap/issues/1272
# This is probably a problem I cant fix on my ona

  


explainer = shap.TreeExplainer(model)
expected_value = explainer.expected_value
if isinstance(expected_value, list):
    expected_value = expected_value[1]
print("Explainer expected value: {expected_value}")



shap.decision_plot(expected_value, shap_values, cat_features)
shap.decision_plot(expected_value, shap_values, features_display, link = 'logit')


# ## Can also use a snapshot thing for catboost so it saves

  


model.save_model('catboost_netflix_data.bin')
model.save_model('catboost_netflix_data.json', format = 'json')

model.load_model('catboost_netflix_data.bin')
print(model.get_params())
print(model.learning_rate_)



# pd.get_dummies(data.type)

# print(data.isnull().sum(axis=0))

# print("distict values:", data['rating'])
# print(data.head(5))

