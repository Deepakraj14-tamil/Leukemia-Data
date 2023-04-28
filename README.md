# Leukemia-Data
    
    Introduction

This code is a Convolutional Neural Network (CNN) based approach to classify leukemia images into four different types of blood cells, namely Eosinophil, Lymphocyte, Monocyte, and Neutrophil.



## Leukemia Diagnosis using Deep Learning Approach

This project aims to develop a deep learning model to diagnose leukemia based on blood cell images. The dataset used in this project is publicly available on Kaggle, and the model was developed using Python programming language with the Keras library.

## Dataset
The dataset used in this project is the ALL-IDB dataset, which contains 108 grayscale images of blood cell samples, 52 of which are ALL (Acute Lymphoblastic Leukemia) positive and 56 of which are negative. The images were obtained from the Department of Hematology of the University of São Paulo, Brazil.

## Methodology
The dataset was split into training, validation, and testing sets, with a ratio of 70:15:15, respectively. The model was developed using a convolutional neural network (CNN) architecture, which has shown great performance in image classification tasks. The model was trained for 50 epochs with a batch size of 16 and an Adam optimizer. 


Installation

To run the code in this project, you will need to install the following libraries:

    TensorFlow
    Keras
    NumPy
    Matplotlib

You can install these libraries using pip:

pip install tensorflow keras numpy matplotlib

Usage

To train the model, run the train.py script:

python train.py

This will train the model using the images in the dataset and save the trained model to a file named leukemia_model.h5.

To test the model, run the test.py script:

python test.py

This will load the trained model from the leukemia_model.h5 file and test it using a set of test images.

## Results
The developed model achieved an accuracy of 82.04% on the testing set, with a precision of 82.7% and a recall of 83.3%. These results show that the model has good performance in diagnosing leukemia based on blood cell images.

## Conclusion
The developed deep learning model has shown promising results in diagnosing leukemia based on blood cell images. However, further research and testing are needed before the model can be used in a clinical setting.

## Credits
This project was developed by [~Deepakraj] as part of [MCA/Part Time Project].

## License
This project is licensed under the [Apache] license.
