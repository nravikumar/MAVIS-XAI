# MAVIS-XAI
XAI benchmarks and framework for XAI selection and best practices protocol

## Image Classification Model with Explainable AI Techniques
This repository contains an image classification model implemented using TensorFlow and Keras API's, along with various explainable AI techniques to visualize and understand the model's decision-making process.
### Table of Contents

1. Model Architecture
2. Dataset
3. Training
4. Explainable AI Techniques
    - Grad-CAM
    - Grad-CAM++
    - Integrated Gradients
    - Saliency Maps
5. Usage
6. Requirements

### Model Architecture
The model is a Convolutional Neural Network (CNN) designed for image classification. It consists of multiple convolutional layers, batch normalization, max pooling, and dropout layers, followed by dense layers for classification. The architecture is as follows:

+ Input shape: (224, 224, 3)
+ Multiple Conv2D layers with ReLU activation and L2 regularization
+ BatchNormalization and MaxPooling2D layers
+ Dropout layers for regularization
+ Dense layers with ReLU activation
+ Final Dense layer with softmax activation for classification

The model is then compiled using SGD optimizer with momentum and categorical crossentropy loss.

### Dataset
The model is trained on a subset of the Tiny ImageNet dataset. The dataset is loaded using keras.utils.image_dataset_from_directory with the following parameters:
+ Image size: 224x224
+ Batch size: 128
+ Label mode: Categorical

### Training 
The model is trained using the following settings:
+ Epochs: 300
+ Validation data: Separate validation set
+ Callbacks: ModelCheckpoint to save the best model based on validation loss
Training progress is visualized using matplotlib to plot accuracy and loss curves.
### Explainable AI Techniques
To understand how the model makes decisions, we've implemented several explainable AI techniques. This will help understand which technique is best for explaining this model's predictions on this dataset.
#### Grad-Cam
Gradient-weighted Class Activation Mapping (Grad-CAM) uses the gradients of any target concept flowing into the final convolutional layer to produce a coarse localization map highlighting important regions in the image for predicting the concept.
#### Grad-Cam++
An improved version of Grad-CAM that provides better visual explanations of CNN model predictions in terms of better object localization and explaining occurrences of multiple object instances in a single image.
#### Integrated Gradients
Integrated Gradients is a technique that attributes the prediction of a deep network to its input features. It involves computing the average gradient while the input varies along a linear path from a baseline to the actual input.
#### Saliency Maps
Saliency Maps highlight the parts of an image that are most important for the model's prediction. They are created by computing the gradient of the output with respect to the input image.
### Usage
In the repository there is a python notebook file that you can run on your preferred development environment, and a specific weights.h5 file.
In the notebook file there's a specific line commented out, this is for you to train the model from scratch and tinker with hyperparamenters. Otherwise, if left as it is, you can easily load the weights of the model that was trained previously, and see all the results and explanations.
### Requirements
+ Tensorflow 2.17.0
+ Keras 3.4.1
+ Numpy 1.26.4
+ Pandas 2.2.2
