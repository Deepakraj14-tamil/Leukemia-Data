# Leukemia-Data
    
    Introduction

This code is a Convolutional Neural Network (CNN) based approach to classify leukemia images into four different types of blood cells, namely Eosinophil, Lymphocyte, Monocyte, and Neutrophil.

The dataset used in this project is the Blood Cell Image Dataset that can be found on Kaggle, which contains two folders, TRAIN and TEST.

This implementation includes data preprocessing, model creation, training, and testing, and provides an accuracy of around 98% on the test set.
Requirements

    Python 3.x
    Numpy
    Matplotlib
    OpenCV
    Seaborn
    Pandas
    Tqdm
    Scikit-learn
    Keras

Usage

    Clone this repository or download the files.
    Ensure that all the required libraries are installed.
    Run leukemia_diagnosis.py to train the model and test it on the test set.
    Optionally, adjust the hyperparameters in the code and train the model again.
    The trained model can be saved as an h5 file by adjusting the checkpoint variable in the code.
    The saved model can be loaded and used for predicting new images.

Acknowledgements

This project uses the Blood Cell Image Dataset available on Kaggle, created by Shivam Bansal.
License

The code is licensed under the MIT License. See LICENSE file for more information.
