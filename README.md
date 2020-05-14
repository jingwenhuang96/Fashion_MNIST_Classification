# Fashion_MNIST_Classification

## Motivation for the project

In this blog, we will use CRISP-DM methodology to know which model is better for image classification with the Fashion MNIST dataset. Classifiers include: 1) traditional machine-learning methods, 2) shallow neural network and 3) deep neural network.

## What is CRISP-DM methodology?
CRISP-DM stands for cross-industry process for data mining, and it includes six steps: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment

## Content
### 1. Business Understanding

Since image classification applications are almost everywhere, knowing which kind of models is more productive can enhance the efficiency of data scientist to build image classification model. In this blog, we would like to know the performance of each classic model for classification, and to be specific, we use the Fashion MNIST dataset to evaluate the performance of different models.

Here are our questions for different model:

How do (1) traditional machine-learning methods (Logistic Regression and Random Forest), (2)shallow neural network (Artificial Neural Network)and (3)deep neural network (Convolutional Neural Network) perform on Fashion MNIST classification?

### 2. Data Understanding

Fashion-MNIST is a dataset of Zalando’s article images — consisting of 70,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

The Fashion-MNIST is to serve as a replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

There are 10 classes in this dataset, below are the labels and description:
0 : T-shirt/top; 1: Trouser; 2 : Pullover; 3 : Dress; 4 : Coat; 5 : Sandal; 6 : Shirt; 7 : Sneaker; 8: Bag; 9 : Ankle boot

### 3. Data Preparation

To train the different model, we have to random split the 70,000 samples into three dataset: training dataset is to train the model, validation dataset is to optimize the parameter of each model, and testing dataset is to test the accuracy rate of our trained model

### 4. Modeling & Evaluation

Here we build four models in four code files: Fashion_MNIST_LR, Fashion_MNIST_RF, Fashion_MNIST_ANN and Fashion_MNIST_2DCNN. 
In each file, we build model and evaluate model using accuracy rate and confusion matrix.

#### Fashion_MNIST_LR

This file uses the logistic regression to classify the images. This files includes the steps of loading data, normalizing data, defining model, evaluating model and creating the confusion matrix. 

**Question:** How does the Logistic Regression model perform on classifying the images of fashion ? Which category does this model make more mistakes?

**Result:**

_Evaluating model:_ 
classification accuracy on test set is 0.8414285714285714

_Confusion matrix:_ 
Actual label 6: label 6 — Shirt is easier to be classified as label 0 — T-shirt/top, label 2 — Pullover, and label 4 — Coat; Actual label 2: label 2 — Pullover is easier to be classified as label 4 — Coat; Logistic Regression model performs relatively not well on classifying shirt.

#### Fashion_MNIST_RF

This file uses the random forest to classify the images. This files includes the steps of loading data, normalizing data, defining model, hyper-parameter tuning, evaluating the model with best depth and creating the confusion matrix. 

**Question:** How does the Fandom Forest Classifier perform on classifying the images of fashion ? Which category does this model make more mistakes?

**Result:**

_Evaluating model:_ 
classification accuracy on test set is: 0.8717857142857143

_Confusion matrix:_ 
label 6 — Shirt is easier to be classified as label 0 — T-shirt/top and label 2 — Pullover； Random Forest model performs relatively not well on classifying shirt.

#### Fashion_MNIST_ANN

This file uses the ANN to classify the images. This files includes the steps of loading data, normalizing data,defining model with different loss function, training the models, evaluating the model with best depth and creating the confusion matrix. 

**Question:** How does the shadow ANN model perform on classifying the images of fashion ? Which category does this model make more mistakes?

**Result:**

_Evaluating model:_ 
for sparse_categorical_crossentropy: Test accuracy: 0.8970714211463928; 
for categorical_crossentropy: Test accuracy: 0.8812857270240784

_Confusion matrix:_ 
There is no significant misclassification between two categories. ANN model performs relatively not well on classifying shirt.

#### Fashion_MNIST_2DCNN

This file uses 2D CNN to classify the images. This files includes the steps of loading data, normalizing data, defining model, training the models, evaluating the model with best depth and creating the confusion matrix. 

**Question:** How does the 2D CNN model perform on classifying the images of fashion ? Which category does this model make more mistakes?

**Result:**

_Evaluating model:_
classification accuracy on test set is: 0.9024285674095154

_Confusion matrix:_ 
label 6 — Shirt is easier to be classified as label 0 — T-shirt/top.
CNN model performs relatively not well on classifying shirt.

## Future Improvement

Keep working on the models to improve the accuracy rate, and adding more classifiers to solve this problem

## Blog of this project in Medium showing the result of this project

https://medium.com/@forever.aiqi/explore-image-classification-with-fashion-mnist-dataset-e375a2483fb7
