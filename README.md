# MAVIS-XAI
XAI benchmarks and framework for XAI selection and best practices protocol

## Image Classification Model with Explainable AI Techniques
This repository contains an image classification model implemented using TensorFlow and Keras API's, along with various explainable AI techniques to visualize and understand the model's decision-making process.
### Table of Contents


1. [Model Architecture](#model-architecture)
2. [Dataset](#dataset)
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
  
To download the dataset use this link: https://www.image-net.org/index.php and register on the website. We've used the Tiny Imagenet dataset for simplicity, using larger and more detailed images would probably increase the accuracy of the model, but increase training time.
### Training 
The model was trained using the following settings:
+ Epochs: 1000
+ Validation data: Separate validation set
+ Callbacks: ModelCheckpoint to save the best model based on validation loss
Training progress is visualized using matplotlib to plot accuracy and loss curves.
To train the model just download the saved weights and the tiny imagenet dataset, a function was used to reorganise the tiny imagenet validation dataset directory to allow for easier training.
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

1. Download the Model's weights "tiny_imagenet_model.checkpoint.weights.h5".
2. Open the notebook file "fullModelAndExplanations.ipynb" using your favourite IDE.
3. Open the link to the dataset and request access to the imagenet libray.
4. Download the tiny imagenet dataset.
5. Decide if you'll use the previously trained model (skip to bullet point 12) or retrain the model from scratch 
6. Decide how many classes to use in the classification, we used 5 (n01443537, n01629819, n01644900, n01770393, n01855672) but there are 200 classes to choose from.
7. Use the "reorganiseValidation.py" script to order all of the validation images in a directory where all the validation images from the same class are in the same directory.
8. Now you can follow the notebook instructions, but the following will explain what you'll do.
9. Download the necessary libraries.
10. Retrain the model with the newly organised validation set.
11. Show the history of the training using pandas.
12. Load the model if not trained from scratch, to load the model you'll need to create the model architecure so make sure to run that cell first.
13. Show predicted class of a specific image chosen and confidence value
14. Show Grad-Cam and Grad-Cam implementations and results
15. Show Integrated Gradients implementations and results
16. Show Saliency maps implementations and results
### Requirements
+ Tensorflow 2.17.0
+ Keras 3.4.1
+ Numpy 1.26.4
+ Pandas 2.2.2
+ Matplotlib 3.9.1
