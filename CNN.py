'''
Nabaa Al Kassab
Class: CS 677
Date: Dec 16, 2021
Final Project
Classifying Images Using Convoluted Neural Networks
'''

import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,MaxPooling2D,Flatten
from keras.utils import np_utils
import numpy as np


#note: code was implemented from 
#https://www.kaggle.com/gpreda/cnn-with-tensorflow-keras-for-fashion-mnist


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

#Reshaping and normalizing the images to feed into model.
trainX = X_train.reshape((X_train.shape[0], 28, 28, 1))
testX = X_test.reshape((X_test.shape[0], 28, 28, 1))
trainX = trainX/255
testX = testX/255

#change labels to integers
trainY = np_utils.to_categorical(Y_train,10)
testY = np_utils.to_categorical(Y_test,10)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#Building the model with Keras layers.
# Use 2 Conv2D layers with 32 & 64 filters 
# each followed by a Max Polling layer of size 2 x 2.
# Using relu activation function. A dropout regularization is also added
# layer with ratio 0.2 to prevent any overfitting.
# Finally the model has a softmax layer with the 10 required classes as output.

classifier = Sequential()
classifier.add(Conv2D(filters=32, kernel_size=(3,3)\
  ,strides=(1, 1), input_shape=(28,28,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(filters=64, kernel_size=(3,3)\
  ,strides=(1, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dropout(rate=0.2))
classifier.add(Dense(units=10, activation='softmax'))

#compile the classifier
classifier.compile(optimizer='adam',\
  loss='categorical_crossentropy',metrics=['accuracy'])


#classifier.summary()
#start tracking the time it takes to use this algorithm
start_time = time.time()
#now train the model:
history = classifier.fit(trainX, trainY,batch_size=128, epochs=20,\
 verbose=2)

predictions = classifier.predict(testX)
predicted_classes = np.argmax(predictions, axis = 1)

#check the accuracy of the model when using test data:
score = classifier.evaluate(testX, testY, verbose=0)
CNN_time = round(time.time()-start_time,2)



print('Test accuracy:', score[1])
print("The time it takes for this algorithm to process is:",CNN_time)