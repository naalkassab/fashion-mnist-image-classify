'''
Nabaa Al Kassab
Class: CS 677
Date: Dec 16, 2021
Final Project
Saving sample image photos & classifying Images Using KNN
'''

#data is downloaded from:
# https://www.kaggle.com/zalando-research/fashionmnist
#The data is already split into 6000 images for training 
#& 10,000 images for testing.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns

'''This function computes accuracy by 
taking in the predictions array & the true labels array'''
def acc(predictions,true_labels):
    count = 0 
    for i in range(0,len(predictions)):
        if predictions[i] == true_labels[i]:
            count = count + 1
    
    return (count/len(predictions))*100



'''this function takes in train features and labels and the k value
to compute the predicted labels. Predicted labels are returned.'''
def knn_classifier (x_train_data, x_test_data, k, y_train_data):
    
    classifier = KNeighborsClassifier(n_neighbors=k)
    y_train_data = y_train_data.ravel()
    classifier.fit(x_train_data, y_train_data)
    y_pred = classifier.predict(x_test_data)
    return y_pred



#load the data into dataframe 
train_data = pd.read_csv('fashion-mnist_train.csv')
test_data = pd.read_csv('fashion-mnist_test.csv')

#get X values for df_Y_train and df_Y_test:
X_train = train_data.drop(columns = ['label'])
X_train = X_train.values
X_test = test_data.drop(columns = ['label'])
X_test = X_test.values


#get Y values for df_Y_train and df_Y_test:
Y_train = train_data.loc[:,'label'].values
Y_test = test_data.loc[:,'label'].values


'''To show an example of how the data looks like, this following code is used.
the resulting image is pasted in the word document. 
Therefore, this code is doc stringed out since it's not needed for analysis.'''

for i in range(8):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(Y_train[i])
    plt.imshow(X_train[i].reshape(28,28),cmap= plt.get_cmap('gray'))
plt.savefig('Example of Images.png')

#data is scaled to avoid large eucledian distances 
scaler = StandardScaler()
scaler.fit(X_train)
x_train = scaler.transform(X_train)
x_test = scaler.transform(X_test)


#when used for KNN classification
#check which k value provides the most accurate results:
#k = 3, 5, 7, 9, 11
#the time each computation takes will be recorded 

start_time = time.time()
predictions_k3 = knn_classifier (X_train, X_test, 3, Y_train)
acc_k3 = acc(predictions_k3,Y_test)
k3_time = round(time.time() - start_time,2)

start_time = time.time()
predictions_k5 = knn_classifier (X_train, X_test, 5, Y_train)
acc_k5 = acc(predictions_k5,Y_test)
k5_time = round(time.time() - start_time,2)

start_time = time.time()
predictions_k7 = knn_classifier (X_train, X_test, 7, Y_train)
acc_k7 = acc(predictions_k7,Y_test)
k7_time = round(time.time() - start_time,2)

start_time = time.time()
predictions_k9 = knn_classifier (X_train, X_test, 9, Y_train)
acc_k9 = acc(predictions_k9,Y_test)
k9_time = round(time.time() - start_time,2)

start_time = time.time()
predictions_k11 = knn_classifier (X_train, X_test, 11, Y_train)
acc_k11 = acc(predictions_k11,Y_test)
k11_time = round(time.time() - start_time,2)


accuracies = {'k = 3': acc_k3,'k = 5': acc_k5,\
    'k = 7': acc_k7,'k = 9': acc_k9, 'k = 11': acc_k11}

comp_times = {'k = 3': k3_time,'k = 5': k5_time,\
    'k = 7': k7_time,'k = 9': k9_time, 'k = 11': k11_time}

cm = confusion_matrix(Y_test, predictions_k5)
# Creating a dataframe for a array-formatted Confusion matrix
#so it will be easy for plotting.
cm_df = pd.DataFrame(cm,
                     index = ['T-shirt','Trouser','Pullover','Dress','Coat'\
                              ,'Sandal','Shirt','Sneaker','Bag','Ankle Boot'], 
                     columns = ['T-shirt','Trouser','Pullover','Dress','Coat'\
                              ,'Sandal','Shirt','Sneaker','Bag','Ankle Boot'])


#Plotting the confusion matrix
plt.figure(figsize=(10,10))
plt.autoscale()
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix for KNN=5 -No feature removal.png')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.savefig('KNN Confusion Matrix-No feature removal.png')
plt.show()

        
print('The accuracies using differnet K Nearest Neighbour classifier:\n')
print(accuracies)
print('/n The time each computation took is:\n')
print(comp_times)


#X_train and X_test will change but Y_train still applies 
remove_cols = ['label','pixel320','pixel321','pixel322','pixel323','pixel324'\
               ,'pixel325','pixel348','pixel349','pixel350','pixel351'\
                   ,'pixel352','pixel353','pixel376','pixel377'\
                       ,'pixel378','pixel379','pixel380','pixel381'\
                           ,'pixel404','pixel405','pixel406','pixel407'\
                               ,'pixel408','pixel409','pixel432','pixel433'\
                                   ,'pixel434','pixel435','pixel436'\
                                       ,'pixel437','pixel460','pixel461',\
                                       'pixel462','pixel463','pixel464',\
                                           'pixel465']


    
    
X_train_no_features = train_data.drop(columns = remove_cols)
X_train_no_features = X_train_no_features.values
X_test_no_features = test_data.drop(columns = remove_cols)
X_test_no_features = X_test_no_features.values



scaler = StandardScaler()
scaler.fit(X_train_no_features)
X_train_no_features = scaler.transform(X_train_no_features)
X_test_no_features = scaler.transform(X_test_no_features) 


start_time = time.time()
classifier = KNeighborsClassifier(n_neighbors=5)
y_train_data = Y_train.ravel()
classifier.fit(X_train_no_features, y_train_data)
y_pred_nof = classifier.predict(X_test_no_features)




acc_k5_nof = acc(y_pred_nof,Y_test)
k5_time = round(time.time() - start_time,2)



cm2 = confusion_matrix(Y_test, y_pred_nof)



#Plotting the confusion matrix
plt.figure(figsize=(10,10))
plt.autoscale()
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix for KNN Classifier-With feature removal')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.savefig('KNN Confusion Matrix-With feature removal.png')
plt.show()




print("The accuracy of classification using KNN",
      "and with removing some features is:\n")
print(acc_k5_nof)
print('The computation time is:\n')
print(k5_time)






