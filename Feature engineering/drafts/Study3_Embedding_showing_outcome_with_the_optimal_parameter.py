import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten
from keras.regularizers import l1
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import Lasso
#import fiel
File_read = pd.read_csv('data.csv', header=None)
feature_sets = File_read.to_numpy()
#set labels
labels = np.array([0] * 100 + [1] * 80)
#set Group for KFold
K= KFold(18)
#reate  models

y_predict_svm_embed    = []
y_predict_svm_no_embed = []
y_predict_knn_embed    = []
y_predict_knn_no_embed = []
svm_classifier = SVC(kernel='linear')
knn_classifier = KNeighborsClassifier(n_neighbors=6)
svm_without_embed = SVC(kernel = 'linear')
knn_without_embed = KNeighborsClassifier(n_neighbors=6)
lasso_model = Lasso(alpha = 0.1)
#Kford
for train_index, test_index in K.split(feature_sets):

    X_train, X_test = feature_sets[train_index], feature_sets[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    # select features
    lasso_model.fit(X_train,y_train)
    feature_weights = lasso_model.coef_
    selected_features = np.where(lasso_model.coef_ != 0)[0]
    X_train_select = X_train[:,selected_features]
    X_test_select  = X_test[:,selected_features]

    # SVM
    
    svm_classifier.fit(X_train_select, y_train)
    y_pred = svm_classifier.predict(X_test_select)
    y_predict_svm_embed = y_predict_svm_embed + list(y_pred)

    # KNN
    
    knn_classifier.fit(X_train_select, y_train)
    y_pred = knn_classifier.predict(X_test_select)
    y_predict_knn_embed = y_predict_knn_embed + list(y_pred)

    # SVM withou embedding
    
    svm_without_embed.fit(X_train,y_train)
    y_pred = svm_without_embed.predict(X_test)
    y_predict_svm_no_embed = y_predict_svm_no_embed + list(y_pred)
    # KNN without embedding
    
    knn_without_embed.fit(X_train,y_train)
    y_pred = knn_without_embed.predict(X_test)
    y_predict_knn_no_embed = y_predict_knn_no_embed + list(y_pred)

#confusion for accuracy ,sensitivity, specificity
confusion_svm = confusion_matrix(labels,np.array(y_predict_svm_embed))
accuracy = (confusion_svm[0,0]+confusion_svm[1,1])/(confusion_svm[0,0]+confusion_svm[0,1]+confusion_svm[1,0]+confusion_svm[1,1])
specificity = confusion_svm[0, 0] / (confusion_svm[0, 0] + confusion_svm[0, 1])
sensitivity = confusion_svm[1, 1] / (confusion_svm[1, 0] + confusion_svm[1, 1])
print('SVM with    Embedding:')
print(f' Accuracy: {accuracy}')
print(f' Sensitivity: {sensitivity}')
print(f' Specificity: {specificity}')
print(' confusion is:\n',confusion_svm)
print("value of confusion: [0,0]->TN,[0,1]->FP,[1,0]->FN,[1,1]_>TP\n")
confusion = confusion_matrix(labels,y_predict_svm_no_embed)
accuracy = (confusion[0,0]+confusion[1,1])/(confusion[0,0]+confusion[0,1]+confusion[1,0]+confusion[1,1])
specificity= confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])
sensitivity = confusion[1, 1] / (confusion[1, 0] + confusion[1, 1])
print('SVM without Embedding:')
print(f' Accuracy: {accuracy}')
print(f' Sensitivity: {sensitivity}')
print(f' Specificity: {specificity}')
print(' confusion is:\n',confusion)
print("value of confusion: [0,0]->TN,[0,1]->FP,[1,0]->FN,[1,1]_>TP\n")
confusion = confusion_matrix(labels,y_predict_knn_embed)
accuracy = (confusion[0,0]+confusion[1,1])/(confusion[0,0]+confusion[0,1]+confusion[1,0]+confusion[1,1])
specificity= confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])
sensitivity = confusion[1, 1] / (confusion[1, 0] + confusion[1, 1])
print('KNN with    Embedding:')
print(f' Accuracy: {accuracy}')
print(f' Sensitivity: {sensitivity}')
print(f' Specificity: {specificity}')
print(' confusion is:\n',confusion)
print("value of confusion: [0,0]->TN,[0,1]->FP,[1,0]->FN,[1,1]_>TP\n")
confusion = confusion_matrix(labels,y_predict_knn_no_embed)
accuracy = (confusion[0,0]+confusion[1,1])/(confusion[0,0]+confusion[0,1]+confusion[1,0]+confusion[1,1])
specificity= confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])
sensitivity = confusion[1, 1] / (confusion[1, 0] + confusion[1, 1])
print('KNN without Embedding:')
print(f' Accuracy: {accuracy}')
print(f' Sensitivity: {sensitivity}')
print(f' Specificity: {specificity}')
print(' confusion is:\n',confusion)
print("value of confusion: [0,0]->TN,[0,1]->FP,[1,0]->FN,[1,1]_>TP\n")
