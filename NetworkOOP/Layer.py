from ActivationType import ActivationType
import numpy as np


class Layer(object):
    def __init__(self, numNeurons, prevNumNeurons, activationFunction, dropOut, isLastLayer=False):
        #layer description
        self.numNeurons=numNeurons
        self.prevNumNeurons=prevNumNeurons
        self.activationFunction=activationFunction
        self.dropOut=dropOut
        self.isLastLayer=isLastLayer       


        #layer parameters
        self.W=np.random.uniform(-0.1,0.1,(self.numNeurons,self.prevNumNeurons))    
        self.B=np.random.uniform(-1,1,(self.numNeurons))
        self.sum=0
        self.delta=np.zeros(self.numNeurons)
        self.a=None #since number of samples in dataset needed as one of the dimensions

        #gradients of parameters
        self.gradW=np.zeros(self.W.shape)
        self.gradB=np.zeros(self.B.shape) #same as delta
        self.gradAF=None;#gradient of activation function
        
    
    def Evaluate(self,input):
        #forward prop
        self.sum=input@self.W.T+self.B #sum is (numSamples x numNeurons)        
        if(self.activationFunction==ActivationType.SIGMOID):
            self.a=self.sigmoid(self.sum)
            self.gradAF=self.a*(1-self.a)
        elif(self.activationFunction==ActivationType.TANH):
            self.a=self.tanh(self.sum)
            self.gradAF=(1-self.a*self.a)
        elif(self.activationFunction==ActivationType.RELU):
            self.a=self.Relu(self.sum)
            self.gradAF=1.0*(self.a>0)
        else: #self.activationFunction==ActivationType.SOFTMAX
            self.a=self.Softmax(self.sum)
            self.gradAF=None #we will use delta directly for back prop
        #apply dropOut unless last layer
        if(self.isLastLayer==False):
            self.mask = np.random.binomial(1, self.dropOut, size=self.a.shape)/self.dropOut
            self.a=self.mask*self.a
            self.gradAF=self.mask*self.gradAF
        
    def clearGradWB(self):
        self.gradW=np.zeros(self.W.shape)
        self.gradB=np.zeros(self.B.shape)

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def Tanh(x):
        return np.tanh(x)

    def Relu(x):
        return np.maximum(x,0)

    def Softmax(x):
        return np.exp(x)/np.sum(np.exp(x))


    



