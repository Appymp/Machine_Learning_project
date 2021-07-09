#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 15:05:33 2021

@author: appanna
"""


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import matplotlib .pyplot as plt
from matplotlib .colors import ListedColormap
from sklearn import neighbors , datasets 
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     StratifiedShuffleSplit)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
 
###Impot dataset
 
food_CO2 = pd.read_csv('Food_Production.csv', delimiter=';')
 
 
###Removing column that we don't need
 
food_CO2.drop(food_CO2.iloc[:,11:], axis = 1, inplace=True)
 
 
####Keeping all features 
X = food_CO2[['Land use change', 'Animal Feed', 'Farm', 'Processing', 'Transport',
              'Packging', 'Retail']].values
 
###Reencoding
 
le = LabelEncoder() 
  
food_CO2['Category_num']= le.fit_transform(food_CO2['Category'])
 
y = food_CO2["Category_num"].astype(int).values
 
 
####Splitting data
 
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=5)
 
 
 
 
#####CLASSIFICATION#######
 
 
 
########K-nearest neigbour###########
 
#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=2)
 
#Train the model using the training sets
knn.fit(X_train, y_train)
 
#Predict the response for test dataset
y_pred = knn.predict(X_test)
 
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
 
####Error
 
error = 1 - knn.score(X_test, y_test)
print('Erreur: %f' % error)
 
####Find the best k
 
 
####Cross Validatation method
 
# Only odd numbers, to prevent ties
param_grid = {'n_neighbors': np.arange(1, 20, 2)}
 
 
knn = KNeighborsClassifier()
 
# Perform grid search with cross-validation
ss = StratifiedShuffleSplit(n_splits=5, test_size=.3, random_state=0)
gscv = GridSearchCV(knn, param_grid, cv=ss)
gscv.fit(X, y)
 
 
print("Best params:", gscv.best_params_)
print("Best score:", gscv.best_score_)
 
 
 
# Print out confusion matrix
cmat = confusion_matrix(y_test, y_pred)
#print(cmat)
print('TP - True Negative {}'.format(cmat[0,0]))
print('FP - False Positive {}'.format(cmat[0,1]))
print('FN - False Negative {}'.format(cmat[1,0]))
print('TP - True Positive {}'.format(cmat[1,1]))
print('Accuracy Rate: {}'.format(np.divide(np.sum([cmat[0,0],cmat[1,1]]),np.sum(cmat))))
print('Misclassification Rate: {}'.format(np.divide(np.sum([cmat[0,1],cmat[1,0]]),np.sum(cmat))))
 
 
# Retrain model using optimal k-value
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
 
# Print out classification report and confusion matrix
print(classification_report(y_test, pred))
 
 
 
########Random Forest#########
 
from sklearn.ensemble import RandomForestClassifier
 
# Feature Scaling
#from sklearn.preprocessing import StandardScaler
 
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)
 
rfc=RandomForestClassifier(random_state=0)
 
param_grid = { 
    'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500]
}
 
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= ss)
CV_rfc.fit(X_train, y_train)
 
 
CV_rfc.best_params_
print("Best score:", CV_rfc.best_score_)
 
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)
 
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)
 
y_pred_clf=clf.predict(X_test)
 
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_clf))
 
 
####Selecting features
 
feature = food_CO2.iloc[:,3:10]
 
feature_imp = pd.Series(clf.feature_importances_,index=feature.columns).sort_values(ascending=False)
feature_imp
 
#Plot
 
import seaborn as sns
 
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
 
 
###Generating model on selected features
 
X_new = food_CO2[['Animal Feed','Farm','Packging']]
 
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y, 
                                                                    test_size=0.30,
                                                                    random_state=5)
 
 
 
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)
 
#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train_new,y_train_new)
 
y_pred_clf=clf.predict(X_test_new)
 
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test_new, y_pred_clf))
 
 
 
#####Naive Bayes######
 
 
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
 
#Create a Gaussian Classifier
gnb = GaussianNB()
 
# Train the model using the training sets
gnb.fit(X_train, y_train)
 
#Predict Output
y_pred_gnb= gnb.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_gnb))
 
 
######## CLUSTERING #########
 
###K means
 
from sklearn.cluster import KMeans
 
 
X_clust = food_CO2.copy()
 
##Keeping all the features
X_clust.drop(columns=['Category', 'Sub_category', 'Food product'], inplace=True)
 
 
#Cluster K-means
 
model=KMeans(n_clusters=4, random_state = 0)
 
#adapter le modèle de données
model.fit(X_clust)
 
# Plotting the cluster centers and the data points on a 2D plane
plt.scatter(X[:, 0], X[:, -1])
    
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', marker='x')
    
plt.title('Data points and cluster centroids')
plt.show()
 
 
# Calculate silhouette_score
from sklearn.metrics import silhouette_score
 
print(silhouette_score(X, model.labels_))
 
 
# Import the KElbowVisualizer method 
from yellowbrick.cluster import KElbowVisualizer
 
# Instantiate a scikit-learn K-Means model
model = KMeans(random_state=0)
 
# Instantiate the KElbowVisualizer with the number of clusters and the metric 
visualizer = KElbowVisualizer(model, k=(2,6), metric='silhouette', timings=False)
 
# Fit the data and visualize
visualizer.fit(X_clust)    
visualizer.poof() 
 
 
#####Performance of our clustering
 
labels = le.fit_transform(food_CO2['Category'])
print(labels)
print(model.labels_)
 
 
##Adjusted Rand Index
metrics.adjusted_rand_score(model.labels_, labels)
#0.06
 
#Fowlkes-Mallows Score
from sklearn.metrics.cluster import fowlkes_mallows_score
fowlkes_mallows_score (labels, model.labels_)
#0.4
 
#Mutual Information Based Score
 
#Normalized Mutual Information (NMI)
from sklearn.metrics.cluster import normalized_mutual_info_score
normalized_mutual_info_score (labels, model.labels_)
#0.2
 
##Adjusted Mutual Information (AMI)
from sklearn.metrics.cluster import adjusted_mutual_info_score
adjusted_mutual_info_score (labels, model.labels_)
#0.08

### Data visualization
import seaborn as sns
sns.set_style('whitegrid')

#swarm plot
sns.swarmplot(data=food_CO2,x='Category', y='Total_emissions')

#Count plot
sns.countplot(x='Category',data=food_CO2)

#Box plot of features
box = pd.melt(food_CO2,id_vars=['Category'], value_vars=['Land use change', 'Animal Feed', 'Farm', 'Processing', 'Transport',
                                    'Packging', 'Retail'])
sns.boxplot(data=box,x='variable', y='value')
