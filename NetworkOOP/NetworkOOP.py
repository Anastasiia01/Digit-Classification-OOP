import sys
import os
import cv2
from Layer import Layer
from NN import NN
from ActivationType import ActivationType
from GradientType import GradientType
import numpy as np

def main():
    train = np.empty((1000,28,28),dtype='float64')
    trainY = np.zeros((1000,10,1))
    test = np.empty((10000,28,28),dtype='float64')
    testY = np.zeros((10000,10,1))

    # Load in the images
    i = 0
    for filename in os.listdir('C:/Users/anast/Documents/Deep Learning/Data/Training1000/'):
        y = int(filename[0])
        trainY[i,y] = 1.0
        train[i] = cv2.imread('C:/Users/anast/Documents/Deep Learning/Data/Training1000/{0}'.format(filename),0)/255.0 # 0 flag stands for greyscale; for color, use 1
        i = i + 1
    i = 0 # read test data
    for filename in os.listdir('C:/Users/anast/Documents/Deep Learning/Data/Test10000/'):
        y = int(filename[0])
        testY[i,y] = 1.0
        test[i] = cv2.imread('C:/Users/anast/Documents/Deep Learning/Data/Test10000/{0}'.format(filename),0)/255.0
        i = i + 1
    trainX = train.reshape(train.shape[0],train.shape[1]*train.shape[2],1)
    testX = test.reshape(test.shape[0],test.shape[1]*test.shape[2],1)

    network=NN(trainX,trainY,[50,10],ActivationType.SIGMOID, ActivationType.SIGMOID,GradientType.STOCHASTIC)

    




if __name__ == "__main__":
    sys.exit(int(main() or 0))
