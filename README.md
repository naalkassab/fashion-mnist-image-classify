# fashion-mnist-image-classify

A data set consisting of 16,000 images of different articles of clothing is trained and tested with different classification algorithms. 
Each image in the dataset is 28 by 28 pixels. Each pixel has an integer value between 0 to 255 that determines the lightness or darkness of the pixel. There is a total of 784 pixels (features) in this data set. 
The goal of this project is to determine how accurate K Nearest Neighbours (KNN), Decision Trees, Random Forest, and Naïve Bayes classifiers are in classifying the images according to their label. 
This is then compared to an image processing algorithm, Convolutional Neural Networks (CNN). 
The following labels are used, and below is an example of how the images look like when plotted in python. 
The hypothesis that pixels located in the centre of each image have minimal impact on accuracy is explored. 
It is likely that the classifiers rely on the outline of each figure (e.g., shape of a sneaker) to perform the classification. 


Note: each file can be run in Python independently and accuracies will be printed.


<img width="857" alt="image" src="https://user-images.githubusercontent.com/47003750/192933885-9d210ec0-234a-4416-ab7e-bff239973b61.png">

# Description of Files

**Randomforest.py:** The Random Forest Classifier is used and different hyperparameters (tree depth and number of trees) are tested to determine the best accuracy or least error rate as seen the in graph generated by the Python file.


**Bayes.py:** Naive Bayes is used in this file. This classifier calculates the probability of class given all its features. It typically benefits from being a very quick algorithm since it assumes features are independent.


**Dectree.py:** This python files uses the decision tree classifier. This is a very simple classifier with minimal preprocessing but typically longer training time.


**KNN.py:**K nearest neighbour algorithm is used. The data was scaled prior to training to avoid domination by scaling factors. Hyperparameters were chosen based on computation of accuracy. K=5 produced the best accuracy as seen below:

<img width="838" alt="image" src="https://user-images.githubusercontent.com/47003750/192934266-da84d9da-803f-4de9-a79f-ba30f9b92704.png">

For all files mentioned above, a confusion matrix is displayed for training and predictions using given the data set, as well as for the data set with some features removed. All graphs and images will be saved in the Project folder when you run the files.

**CNN.py:** Convolutional Neural Networks are used to classify the data in this file. Different layers or matrices are used to detect patterns in the images.
The layers are arranged so that they detect simpler patterns first (lines, curves, etc.) and more complex patterns further along.  
This algorithm has the best accuracy compared to all other models as it is commonly used for image processing and classification. 

