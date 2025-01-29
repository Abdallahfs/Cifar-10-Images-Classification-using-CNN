# CIFAR-10 Image Classification 🖼️🔍

## Objective  
Develop a deep learning model using **Convolutional Neural Networks (CNNs)** to classify images from the **CIFAR-10 dataset** into ten distinct categories.

## Project Overview  
This project focuses on leveraging **CNNs** to recognize and classify images into their respective categories. The CIFAR-10 dataset consists of small images categorized into ten classes, making it a great benchmark for image classification tasks.

## Methodology  

1. **Data Preparation**:  
   - Loaded and preprocessed the CIFAR-10 dataset.  
   - Normalized pixel values to improve training efficiency.  
   - Performed **data augmentation** (rotation, flipping, shifting) to enhance model generalization.  

2. **Model Development**:  
   - Built a **CNN architecture** with convolutional, pooling, and dense layers.  
   - Experimented with different activation functions and optimizers.  
   - Implemented **dropout** and **batch normalization** to prevent overfitting.  

3. **Training and Optimization**:  
   - Split the data into **training (50,000) and testing (10,000) images**.  
   - Trained the model using the **Adam optimizer** and **categorical cross-entropy loss**.  
   - Fine-tuned hyperparameters such as learning rate, batch size, and number of filters.  

4. **Model Evaluation**:  
   - Assessed model performance using **accuracy**.  


## Tools and Technologies  
- Python, TensorFlow, Keras, NumPy, Matplotlib, Seaborn  

## Usage  
This repository contains the dataset preprocessing, CNN model implementation, and evaluation steps. You can clone the repository and run the Jupyter notebook to train and test the model.
