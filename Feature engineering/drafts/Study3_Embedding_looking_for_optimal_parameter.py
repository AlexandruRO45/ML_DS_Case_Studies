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
#create classifiers

L1_list = [0.01,0.02,0.03,0.04,0.05,0.06,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]

X_X = [mm for mm in range(1,61,1)]

#for SVM
accuracy_list = []
sensitivity_list = []
specificity_list = []
#for KNN
accuracy_list2 = []
sensitivity_list2 = []
specificity_list2 = []
for j in L1_list:
        y_predict_svm_embed    = []
        y_predict_knn_embed    = []
        svm_classifier = SVC(kernel='linear')
        knn_classifier = KNeighborsClassifier(n_neighbors=6)
  
        #create an Embedding model
        lasso_model = Lasso(alpha = j)

        #Kford
        for train_index, test_index in K.split(feature_sets):

            X_train, X_test = feature_sets[train_index], feature_sets[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            
            # feature_selection for train model:
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

        #confusion for accuracy ,sensitivity, specificity
        confusion_svm = confusion_matrix(labels,np.array(y_predict_svm_embed))
        accuracy = (confusion_svm[0,0]+confusion_svm[1,1])/(confusion_svm[0,0]+confusion_svm[0,1]+confusion_svm[1,0]+confusion_svm[1,1])
        specificity = confusion_svm[0, 0] / (confusion_svm[0, 0] + confusion_svm[0, 1])
        sensitivity = confusion_svm[1, 1] / (confusion_svm[1, 0] + confusion_svm[1, 1])
        accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        
        confusion = confusion_matrix(labels,y_predict_knn_embed)
        accuracy = (confusion[0,0]+confusion[1,1])/(confusion[0,0]+confusion[0,1]+confusion[1,0]+confusion[1,1])
        specificity= confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])
        sensitivity = confusion[1, 1] / (confusion[1, 0] + confusion[1, 1])
        accuracy_list2.append(accuracy)
        sensitivity_list2.append(sensitivity)
        specificity_list2.append(specificity)
        '''
        plt.bar(range(len(feature_weights)),abs(feature_weights))
        plt.title(f'Lasso with L1(alpha = {j})')
        plt.ylabel('|w|')
        plt.xlabel('Feature Index')
        plt.show()'''
        
plt.plot(L1_list,accuracy_list,'bo--')
plt.plot(L1_list,sensitivity_list,'go--')
plt.plot(L1_list,specificity_list,'yo--')
plt.legend(['Accuracy','Sensitivity','Specificity'])
plt.title('SVM with embedding')
plt.show()
plt.plot(L1_list,accuracy_list2,'bo--')
plt.plot(L1_list,sensitivity_list2,'go--')
plt.plot(L1_list,specificity_list2,'yo--')
plt.legend(['Accuracy','Sensitivity','Specificity'])
plt.title('KNN with embedding')
plt.show()


