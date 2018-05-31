# Gender-Recognition

This is a SVM (Support Vector Machine) implementation of a gender recognition program. 

## Getting Started

Images are packed into a .npy file as a matrix of pixels. However, you can find the 
original data set [here](http://mivia.unisa.it/datasets/video-analysis-datasets/gender-recognition-dataset/).

### Prerequisites

The code is dependent on several third party libraries. You will need to install them 
in order for the program to work.

[Scikit-learn](http://scikit-learn.org/stable/)<br/>
[OpenCV](https://opencv.org/) <br/>
[Dlib](https://pypi.org/project/dlib/) <br/>
[Numpy](http://www.numpy.org/) <br/>
[Scipy](https://www.scipy.org/) <br/>
[Matplot](https://matplotlib.org/)<br/>
[Imutils](https://pypi.org/project/imutils/)

## Training data
The [GENDER-FERET](http://mivia.unisa.it/datasets/video-analysis-datasets/gender-recognition-dataset/) dataset has been used for training. This training set consists of total of 946 images splitted in two classes of 473 women and  473 men.

Explain how to run the automated tests for this system

## Pre-processing

There exist a wide variety of algorithms that extract features using kernel matrices. However, some of them are very sensitive to light and to objects that faces commonly have such as glasses. I took a different approach and instead used vectors and euclidean distances to extract ratios from the face. I order to accomplish the latter is using dlib to extract landmarks from an individual face (which is basically positions on the face). The features used were first picked up using a rule of thumb and some intuiton on what could differ a female from a male. To estimate what features were important, XGboost does the job. 

## Running training

Currently, the best accuracy using my own features were of 85.55%

## Authors

* **Ricardo Duran**

## License

This project is licensed under the Apache License - see the [LICENSE.md](LICENSE.md) file for details


