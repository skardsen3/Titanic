# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 18:59:23 2020

@author: Ashton
"""

#Learning Random Forest Machine Learning. Guided by https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


#Load data and exclude undersirable columns
train = pd.read_csv('D:\\Python\\Kaggle\\Data\\titanic\\train.csv')
test = pd.read_csv('D:\\Python\\Kaggle\\Data\\titanic\\test.csv')
train = train.drop(['Name','Ticket','Cabin'],1)
test = test.drop(['Name','Ticket','Cabin'],1)



#Remove NA's and resolve issues
trainLong = pd.get_dummies(train)
trainLong = trainLong[trainLong["Age"] >= 0]
trainResults = trainLong['Survived']+1
trainNaked = trainLong.drop('Survived', axis = 1)

headers = list(trainNaked.columns)



#Get test and training data
train_features, test_features, train_labels, test_labels = train_test_split(trainNaked, trainResults, test_size = 0.25, random_state = 42)


#Run the random forest
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)


#Map Errors
mape = 100 * (errors / test_labels)
print(mape)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')



#Find most import variables
importances = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(headers, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


#Run on the test data
testLong = pd.get_dummies(test)
testLong['Age'] = testLong['Age'].fillna(testLong['Age'].mean())
testLong['Fare'] = testLong['Fare'].fillna(testLong['Fare'].mean())
testPredict = rf.predict(testLong)

#Format and Export to CSV
final = pd.DataFrame({'PassengerId':testLong['PassengerId'], 'Survived':np.round(testPredict-1,0)}, columns=['PassengerId','Survived'])
final.to_csv('D:\\Python\\Kaggle\\Data\\titanic\\Results.csv', index=False)
