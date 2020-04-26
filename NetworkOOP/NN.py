from ActivationType import ActivationType
from GradientType import GradientType
from Layer import Layer
import numpy as np
from sklearn.utils import shuffle
from BatchNormMode import BatchNormMode

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

    def Evaluate(self,input,batchNorm=False, batchMode=BatchNormMode.TEST): #returns output of network
        self.listOfLayers[0].Evaluate(input,batchNorm,batchMode)
        for i in range(1,(len(self.listOfLayers)-1)):
            self.listOfLayers[i].Evaluate(self.listOfLayers[i-1].a,batchNorm,batchMode)
        #do NOT apply batch Norm on output layer, especially if it is Softmax,
        #since norm reduces the difference between outputs, what softmax is relying on while making desion 
        self.listOfLayers[len(self.listOfLayers)-1].Evaluate(self.listOfLayers[len(self.listOfLayers)-2].a,batchNorm=False)
        return self.listOfLayers[len(self.listOfLayers)-1].a
    
    def Train(self, gradientType, epochsNum=50, alpha=0.1, regLambda=0.01 , batchNorm=False, Size=40):
        samplesNum=self.X.shape[0]
        epsilon=1E-6
        if(gradientType==GradientType.STOCHASTIC):
            batchSize=1
        elif(gradientType==GradientType.MINIBATCH):
            batchSize=Size
        else:#if(gradientType==GradientType.BATCH):
            batchSize=samplesNum
        for j in range(epochsNum):
            self.X,self.Y=shuffle(self.X,self.Y)
            loss=0
            for i in range(0,samplesNum,batchSize):#define batchSize
                #compute forward pass
                xBatch=self.X[i:i+batchSize]
                yBatch=self.Y[i:i+batchSize]
                actualOut=self.Evaluate(xBatch,batchNorm,batchMode=BatchNormMode.TRAIN)
                #print(actualOut)
                if(self.lastLayerAF==ActivationType.SOFTMAX):
                    loss+=-(yBatch*np.log(actualOut+0.001)).sum()
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
                        self.listOfLayers[levelNum].delta=self.listOfLayers[levelNum+1].delta@(self.listOfLayers[levelNum+1].W)*self.listOfLayers[levelNum].gradAF 
                        if(batchNorm==True):
                            self.listOfLayers[levelNum].GradBeta=self.listOfLayers[levelNum].delta.sum(axis=0)
                            self.listOfLayers[levelNum].GradGamma=(self.listOfLayers[levelNum].delta*self.listOfLayers[levelNum].sHat).sum(axis=0)
                            self.listOfLayers[levelNum].deltaBn=(self.listOfLayers[levelNum].delta*self.listOfLayers[levelNum].gamma)/(batchSize*((self.listOfLayers[levelNum].sigma2+epsilon)**0.5))*(batchSize-1-(self.listOfLayers[levelNum].sHat)**2)
                            #self.listOfLayers[levelNum].deltaBn=(self.listOfLayers[levelNum].delta * self.listOfLayers[levelNum].gamma)/(batchSize*np.sqrt(self.listOfLayers[levelNum].sigma2 +self.listOfLayers[levelNum].epsilon )) * (batchSize - 1 - (self.listOfLayers[levelNum].sHat * self.listOfLayers[levelNum].sHat))
                    if(levelNum>0):
                        prevOut=self.listOfLayers[levelNum-1].a
                    else:#level #zero
                        prevOut=xBatch 
                    if(batchNorm==True and self.listOfLayers[levelNum].isLastLayer==False):
                        self.listOfLayers[levelNum].gradB+=self.listOfLayers[levelNum].deltaBn.sum(axis=0)
                        self.listOfLayers[levelNum].gradW+=self.listOfLayers[levelNum].deltaBn.T@prevOut
                    else:
                        self.listOfLayers[levelNum].gradB+=self.listOfLayers[levelNum].delta.sum(axis=0)
                        self.listOfLayers[levelNum].gradW+=self.listOfLayers[levelNum].delta.T@prevOut                        
                    levelNum-=1  
                self.UpdateGradsWB(alpha,regLambda,batchSize,batchNorm)
            print("epoch = ", j, "loss = ", loss)

    def UpdateGradsWB(self, alpha,regLambda,batchSize,batchNorm=False):
        for level in self.listOfLayers:
            level.W=level.W-alpha*(level.gradW/batchSize)-alpha*regLambda*level.W
            level.B=level.B-alpha*(level.gradB/batchSize)
            if(batchNorm==True):
                level.beta=level.beta-alpha*(level.gradBeta/batchSize)
                level.gamma=level.gamma-alpha*(level.gradGamma/batchSize)
            level.clearGrads()

    def GetAccuracy(self, testX, testY, batchNorm=False):
        rightCount=0
        testsNum=testX.shape[0]
        for i in range(testsNum):
            yi=testY[i]
            #forward pass:
            a2=self.Evaluate(testX[i],batchNorm,batchMode=BatchNormMode.TEST)
            maxIdx=np.argmax(a2)
            if(yi[maxIdx]==1):
                rightCount+=1
        accuracy=(rightCount/testsNum)*100
        return accuracy




   


