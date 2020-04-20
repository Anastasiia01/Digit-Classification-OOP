from ActivationType import ActivationType
from GradientType import GradientType
from Layer import Layer
import numpy as np

class NN(object):
    def __init__(self, X,Y,listNeuronsInLayers, activationFunction, lastActivationFunction, gradientType, dropOut=1.0, batchNorm=False):
        self.X=X
        self.Y=Y
        self.listOfLayers=[]
        self.gradientType=gradientType
        #self.dropOut=dropOut
        self.batchNorm=batchNorm
        firstLayer=Layer(listNeuronsInLayers[0], X.shape[1], activationFunction,dropOut,False)
        self.listOfLayers.append(firstLayer)
        for i in range(1,(len(listNeuronsInLayers)-1)):
            layer=Layer(listNeuronsInLayers[i],listNeuronsInLayers[i-1],activationFunction,dropOut,False)
            self.listOfLayers.append(layer)
        lastLayer=Layer(listNeuronsInLayers[len(listNeuronsInLayers)-1], listNeuronsInLayers[len(listNeuronsInLayers)-2], lastActivationFunction,dropOut,False)
        self.listOfLayers.append(lastLayer)

    def Evaluate(self):
        x=0
    def Train(self):
        x=0
    def GetAccuracy(self):
        accuracy=0
        return accuracy
    def Classify(self):
        x=0


   


