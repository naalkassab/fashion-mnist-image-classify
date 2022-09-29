'''
Nabaa Al Kassab
Class: CS 677
Date: Dec 16, 2021
Final Project
Classifying Images Using Random Forest
'''

#data is downloaded from:
# https://www.kaggle.com/zalando-research/fashionmnist
#The data is already split into 6000 images for training 
#& 10,000 images for testing.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
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
'''
for i in range(8):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(Y_train[i])
    plt.imshow(X_train[i].reshape(28,28),cmap= plt.get_cmap('gray'))
plt.show()
'''


#use random forest classifier to make predicitions
'''this method takes in the required depth for the random forest classifier
and returns the error rates for 1 to 10 subtrees'''
def rand_forest(d):
    for i in range(1,11):
        model = RandomForestClassifier(n_estimators=i,max_depth=d,\
                                       criterion='entropy')
        model.fit(X_train,Y_train)
        prediction = model.predict(X_test)
        error_rate = np.mean(prediction!=Y_test)
        if i == 1: 
            d_res = np.array([[i, error_rate]])
        else:
            d_res = np.append(d_res,[[i,error_rate]],axis= 0)
    return d_res

#compute all the depths required
d2 = rand_forest(2)
d3 = rand_forest(3)
d6 = rand_forest(6)
d9 = rand_forest(9)
d10 = rand_forest(10)
d20 = rand_forest(20)

#plot all the depths
x, y = d2.T
plt.plot(x,y,marker='o', label = 'd = 2')
x, y = d3.T
plt.plot(x,y,marker='o', label = 'd = 3')
x, y = d6.T
plt.plot(x,y,marker='o', label = 'd = 6')
x, y = d9.T
plt.plot(x,y,marker='o', label = 'd = 9')
x, y = d10.T
plt.plot(x,y,marker='o', label = 'd = 10')
x, y = d20.T
plt.plot(x,y,marker='o', label = 'd = 20')

#show the graph of how the error rate changes based on 
#the hyperparameter chosen
#this will help determine the best hyperparamter
plt.xlabel("N-Values")
plt.ylabel("Error Rates")
plt.legend()
plt.show()

#now compute accuracy and algorithm run time for the best hyperparameters
start_time = time.time()
model = RandomForestClassifier(n_estimators=10,max_depth=20,\
                                       criterion='entropy')
model.fit(X_train,Y_train)
rand_forest_prediction = model.predict(X_test)
rand_forest_time = time.time()-start_time


#complete confusion matrix:
cm = confusion_matrix(Y_test, rand_forest_prediction)
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
plt.title('Confusion Matrix for Random Forest Classifier-No feature removal')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.savefig('Random Forest Confusion Matrix-No feature removal.png')
plt.show()



print("Based on the graph, using n=10 and a depth of 20",\
    " produces results with the least error.")

print('The accuracy of classification using Random Forest is:\n')
print(round(acc(rand_forest_prediction,Y_test),2))
print('The computation time is:\n')
print(rand_forest_time)


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



start_time = time.time()
model = RandomForestClassifier(n_estimators=10,max_depth=20,\
                                       criterion='entropy')
model.fit(X_train_no_features,Y_train)
rand_forest_prediction_no_features = model.predict(X_test_no_features)
rand_forest_time = time.time()-start_time


RF_acc_no_features = acc(rand_forest_prediction_no_features,Y_test)



cm2 = confusion_matrix(Y_test, rand_forest_prediction_no_features)



#Plotting the confusion matrix
plt.figure(figsize=(10,10))
plt.autoscale()
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix for Random Forest Classifier-With feature removal')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.savefig('Random Forest Confusion Matrix-With feature removal.png')
plt.show()




print("The accuracy of classification using Naive Bayes",
      "and with removing some features is:\n")
print(RF_acc_no_features)
print('The computation time is:\n')
print(rand_forest_time)


