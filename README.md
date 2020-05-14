# Fashion_MNIST_Classification

## Motivation for the project

In this project, the fashion MNIST dataset is used to train the different machine learning methods, including the logistic regression classifier, random forest classifier, artificial neural network and convolutional neural network. The purposes of this project is to understand which models are better in this case

## Blog of this project in Medium showing the result of this project

https://medium.com/@forever.aiqi/explore-image-classification-with-fashion-mnist-dataset-e375a2483fb7

## Libraries used

matplotlib 
numpy
sklearn
tensorflow.keras

## Coding Files

There are four code files in this repository

### Fashion_MNIST_LR

This file uses the logistic regression to classify the images. This files includes the steps of loading data, normalizing data, defining model, evaluating model and creating the confusion matrix. 

Question: How does the Logistic Regression model perform on classifying the images of fashion ? Which category does this model make more mistakes?

Result:
Evaluating model: classification accuracy on test set is 0.8414285714285714
Confusion matrix:  Actual label 6: label 6 — Shirt is easier to be classified as label 0 — T-shirt/top, label 2 — Pullover, and label 4 — Coat; Actual label 2: label 2 — Pullover is easier to be classified as label 4 — Coat; Logistic Regression model performs relatively not well on classifying shirt.

### Fashion_MNIST_RF

This file uses the random forest to classify the images. This files includes the steps of loading data, normalizing data, defining model, hyper-parameter tuning, evaluating the model with best depth and creating the confusion matrix. 

Question: How does the Fandom Forest Classifier perform on classifying the images of fashion ? Which category does this model make more mistakes?

Result:
Evaluating model: classification accuracy on test set is: 0.8717857142857143
Confusion matrix: label 6 — Shirt is easier to be classified as label 0 — T-shirt/top and label 2 — Pullover； Random Forest model performs relatively not well on classifying shirt.

### Fashion_MNIST_ANN

This file uses the ANN to classify the images. This files includes the steps of loading data, normalizing data,defining model with different loss function, training the models, evaluating the model with best depth and creating the confusion matrix. 

Question: How does the shadow ANN model perform on classifying the images of fashion ? Which category does this model make more mistakes?

Result:
Evaluating model: 
for sparse_categorical_crossentropy: Test accuracy: 0.8970714211463928; 
for categorical_crossentropy: Test accuracy: 0.8812857270240784
Confusion matrix: 
There is no significant misclassification between two categories. ANN model performs relatively not well on classifying shirt.

### Fashion_MNIST_2DCNN

This file uses 2D CNN to classify the images. This files includes the steps of loading data, normalizing data, defining model, training the models, evaluating the model with best depth and creating the confusion matrix. 

Question: How does the 2D CNN model perform on classifying the images of fashion ? Which category does this model make more mistakes?

Result:
Evaluating model: 
classification accuracy on test set is: 0.9024285674095154
Confusion matrix: 
label 6 — Shirt is easier to be classified as label 0 — T-shirt/top.
CNN model performs relatively not well on classifying shirt.

## Future Improvement

Keep working on the models to improve the accuracy rate, and adding more classifiers to solve this problem

