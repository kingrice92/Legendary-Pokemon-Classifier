#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 03:21:03 2022

@author: kingrice
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn import metrics 

dataset = pd.read_csv('Pokemon.csv')

# Replace the NANs in the data with empty strings
dataset = dataset.replace(np.nan,'', regex=True)

# Split dataset in features and target variable
features = dataset.columns[2:-1].to_numpy()
X = dataset[features]
y = dataset['Legendary']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Since the pokemon dataset include both continuous and categorical data, the 
# categorical data needs to be encoded as numbers so that it can be handled 
# by the tree classifier
ohe = OneHotEncoder(handle_unknown='ignore')
ohe = ohe.fit(X_train)

X_train_ohe = ohe.transform(X_train).toarray()

# Collecte the output of the encoder into a dataset. These headers will be
# used to label the resulting decision tree graph
ohe_df = pd.DataFrame(X_train_ohe, columns=ohe.get_feature_names(X_train.columns))

# Initialize and train the decision tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train_ohe, y_train)

# Plot the resulting decision tree
tree.plot_tree(clf, feature_names = ohe_df.columns, class_names=np.unique(y).astype('str'), filled = True)
plt.show()

# Predict the response for test dataset and score it's accuracy
X_test_ohe = ohe.transform(X_test)
y_preds = clf.predict(X_test_ohe)

print('Model Accuracy: ', metrics.accuracy_score(y_test, y_preds))

# Include some tests to confirm model
test1 = bool(clf.predict(ohe.transform([['Dark','Flying',680,126,131,95,131,98,99,6]])))
print('Is Yveltal (#717) a legendary Pokemon? ', str(test1))
test2 = bool(clf.predict(ohe.transform([['Fire','',580,115,115,85,90,75,100,2]])))
print('Is Entei (#244) a legendary Pokemon? ', str(test2))
test3 = bool(clf.predict(ohe.transform([['Water','',314,44,48,65,50,64,43,1]])))
print('Is Squirtle (#7) a legendary Pokemon? ', str(test3))
test4 = bool(clf.predict(ohe.transform([['Steel','Psychic',300,57,24,86,24,86,23,4]])))
print('Is Bronzor (#436) a legendary Pokemon? ', str(test4))

# Export figure as dot file (better resuloution than saving from python)
tree.export_graphviz(clf, out_file='tree.dot', feature_names = ohe_df.columns,
                class_names = np.unique(y).astype('str'), rounded = True, 
                proportion = False, filled = True)

#Type dot -Tpng tree.dot -o tree.png into the command line to convert dot to png