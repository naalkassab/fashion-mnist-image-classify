'''
Nabaa Al Kassab
Class: CS 677
Date: Dec 16, 2021
Final Project
Classifying Images Using Decision Tree
'''

#data is downloaded from:
# https://www.kaggle.com/zalando-research/fashionmnist
#The data is already split into 6000 images for training 
#& 10,000 images for testing.

import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import tree
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


#use decision classifier to make predicitions
print("Please give this file sometime to load as it takes a while...")
start_time = time.time()
model = tree.DecisionTreeClassifier(criterion='entropy')
model.fit(X_train,Y_train)
prediction = model.predict(X_test)
dectree_time = time.time()-start_time


cm = confusion_matrix(Y_test, prediction)
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
plt.title('Confusion Matrix for Decision Tree Classifier-No feature removal')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.savefig('Decision Tree Confusion Matrix-No feature removal.png')
plt.show()


print('The accuracy of classification using Decision Tree is:\n')
print(round(acc(prediction,Y_test),2))
print('The computation time is:\n')
print(dectree_time)


'''Now some features will be removed and 
the algorithms accuracy will be checked again '''

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
model = tree.DecisionTreeClassifier(criterion='entropy')
model.fit(X_train_no_features,Y_train)
prediction_no_features = model.predict(X_test_no_features)
dectree_time_2 = time.time()-start_time


cm2 = confusion_matrix(Y_test, prediction_no_features)



#Plotting the confusion matrix
plt.figure(figsize=(10,10))
plt.autoscale()
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix for Decision Tree Classifier-With feature removal')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.savefig('Decision Tree Confusion Matrix-With feature removal.png')
plt.show()




print("The accuracy of classification using Decision Tree",
      "and with removing some features is:\n")
print(round(acc(prediction_no_features,Y_test),2))
print('The computation time is:\n')
print(dectree_time_2)
