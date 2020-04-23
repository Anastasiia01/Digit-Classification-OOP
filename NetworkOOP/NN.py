from ActivationType import ActivationType
from GradientType import GradientType
from Layer import Layer
import numpy as np
from sklearn.utils import shuffle

class NN(object):
    def __init__(self, X,Y,listNeuronsInLayers, activationFunction, lastActivationFunction, dropOut=1.0):
        self.X=X
        self.Y=Y
        self.listOfLayers=[]
        self.AF=activationFunction
        self.lastLayerAF=lastActivationFunction
        firstLayer=Layer(listNeuronsInLayers[0], X.shape[1], activationFunction,dropOut,False)
        self.listOfLayers.append(firstLayer)
        for i in range(1,(len(listNeuronsInLayers)-1)):
            layer=Layer(listNeuronsInLayers[i],listNeuronsInLayers[i-1],activationFunction,dropOut,False)
            self.listOfLayers.append(layer)
        lastLayer=Layer(listNeuronsInLayers[len(listNeuronsInLayers)-1], listNeuronsInLayers[len(listNeuronsInLayers)-2], lastActivationFunction,dropOut,True)
        self.listOfLayers.append(lastLayer)
        #self.Evaluate()

    def Evaluate(self,input): #returns output of network
        self.listOfLayers[0].Evaluate(input)
        for i in range(1,len(self.listOfLayers)):
            self.listOfLayers[i].Evaluate(self.listOfLayers[i-1].a)
        return self.listOfLayers[len(self.listOfLayers)-1].a

    def Train(self, gradientType, epochsNum=100, alpha=0.1, regLambda=0.1, batchNorm=False, batchSize=50):
        samplesNum=self.X.shape[0]
        for j in range(epochsNum):
            self.X,self.Y=shuffle(self.X,self.Y)
            loss=0
            for i in range(0,samplesNum,batchSize):#define batchSize
                #compute forward pass
                xBatch=self.X[i:i+batchSize]
                yBatch=self.Y[i:i+batchSize]
                actualOut=self.Evaluate(xBatch)
                if(self.lastLayerAF==ActivationType.SOFTMAX):
                    loss+=-(yBatch*np.log(actualOut)).sum()
                else:
                    loss+=((actualOut-yBatch)*(actualOut-yBatch)).sum()/2
                #backpropagation: updating weigths and biases
                levelNum=len(self.listOfLayers)-1
                while(levelNum>=0):
                    if(self.listOfLayers[levelNum].isLastLayer==True):#last layer
                        if(self.lastLayerAF==ActivationType.SOFTMAX):
                            self.listOfLayers[levelNum].delta=(actualOut - yBatch)
                        else:
                            self.listOfLayers[levelNum].delta=(actualOut - yBatch)*self.listOfLayers[levelNum].gradAF
                    else:#regular layer
                        #print(self.listOfLayers[levelNum+1].delta.shape)
                        #print(self.listOfLayers[levelNum+1].W.shape)
                        self.listOfLayers[levelNum].delta=self.listOfLayers[levelNum+1].delta@(self.listOfLayers[levelNum+1].W)*self.listOfLayers[levelNum].gradAF                    
                    if(levelNum>0):
                        prevOut=self.listOfLayers[levelNum-1].a
                    else:#level #zero
                        prevOut=xBatch
                    #print(self.listOfLayers[levelNum].gradB.shape)
                    #print(self.listOfLayers[levelNum].delta.shape)
                    self.listOfLayers[levelNum].gradB+=self.listOfLayers[levelNum].delta.sum(axis=0)
                    self.listOfLayers[levelNum].gradW+=self.listOfLayers[levelNum].delta.T@prevOut 
                    levelNum-=1                
                if(gradientType==GradientType.STOCHASTIC):
                    self.UpdateGradsWB(alpha,regLambda)
                if(gradientType==GradientType.MINIBATCH):
                    self.UpdateGradsWB(alpha,regLambda,batchSize)
            if(gradientType==GradientType.BATCH):
                self.UpdateGradsWB(alpha,regLambda,samplesNum)
            print("epoch = ", j, "loss = ", loss)

    def UpdateGradsWB(self, alpha,regLambda,batchSize=1):
        for level in self.listOfLayers:
            level.W=level.W-alpha*(level.gradW/batchSize)-alpha*regLambda*level.W
            level.B=level.B-alpha*(level.gradB/batchSize)
            level.clearGradWB()

    def GetAccuracy(self):
        accuracy=0
        return accuracy

    def Classify(self):
        x=0


   


