'''
Originally written in a jupyter notebook
Used more visualizations from seaborn and matplotlib originally
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#load data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

#create pandas dataframe
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])

#make seaborn lmplot that shows regression lines for the cancerous/benign tumors
#target = 0 means cancerous
sns.lmplot('mean area', 'mean radius', df, hue = 'target')

#adding target column to dataframe
df['target'] = cancer['target']

#splitting training data
from sklearn.model_selection import train_test_split
#X is all info except the target column, and y is the target or classification
X = df.drop('target', axis = 1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

#setting grid search as a means of finding best parameters for SVM algorithm
from sklearn.model_selection import GridSearchCV
#dictionary with values for each parameter every power of 10
param_grid = {'C':[0.1,1,10,100,100],'gamma':[1,0.1,0.01,0.001,0.0001]}
#create grid object and search, verbose prints activity to moniter status
grid = GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train, y_train)

#printing the selected parameters, C of 10 and gamma of 0.0001
grid.best_params_

#creating predictions
grid_predictions = grid.predict(X_test)

#printing of classification report
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))

'''
Classification report has a .95 precision, recall, and f1-score
Would go deeper and more precise with grid search if this dataset had more data points
'''
