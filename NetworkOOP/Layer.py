from ActivationType import ActivationType
from BatchNormMode import BatchNormMode
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
        self.delta=None
        self.a=None #since number of samples in dataset needed as one of the dimensions

        #gradients of parameters
        self.gradW=np.zeros(self.W.shape)
        self.gradB=np.zeros(self.B.shape) #same as delta
        self.gradAF=None;#gradient of activation function
        
        #Batch Normalization parameters
        #s(sum) -> sHat -> sb -> activation func
        self.mu=np.zeros(self.numNeurons)#mu is 1 x numNeurons
        self.muRunning=np.zeros(self.numNeurons)
        self.sigma2=np.zeros(self.numNeurons)
        self.sigma2Running=np.zeros(self.numNeurons)
        self.epsilon=1E-6
        self.sHat=None#sHat is (numSamples x numNeurons), where S is centered around the mean, scaled by varience
        
        self.sb=None#sb is (numSamples x numNeurons), where sHat is scaled by gamma and shifted by beta
        self.gamma=np.random.uniform(-0.1,0.1,(self.numNeurons))
        self.beta=np.random.uniform(-1,1,(self.numNeurons))
        self.gradGamma=np.zeros(self.numNeurons)
        self.gradBeta=np.zeros(self.numNeurons)
        self.deltaBn=None#deltabn is (numSamples x numNeurons)  #grad of S

        #Adam learning rate optimization parameters
        self.mW=0
        self.vW=0
        self.mB=0
        self.vB=0
    
    def Evaluate(self,input, batchNorm=False, batchMode=BatchNormMode.TRAIN):
        #forward prop
        self.S=input@self.W.T+self.B #S is (numSamples x numNeurons)  
        if(batchNorm==True):
            #s(sum) -> sHat -> sb -> activation func
            if(batchMode==BatchNormMode.TRAIN):
                self.mu=np.mean(self.S,axis=0)
                self.sigma2=np.var(self.S,axis=0)
                self.muRunning=0.9*self.muRunning+(1-0.9)*self.mu
                self.sigma2Running=0.9*self.sigma2Running+(1-0.9)*self.sigma2
            else:   
                self.mu=self.muRunning
                self.sigma2=self.sigma2Running
            self.sHat=(self.S-self.mu)/((self.sigma2+self.epsilon)**0.5)
            self.sb=self.sHat*self.gamma+self.beta
            sum=self.sb
        else:#without Batch Normalization:
            sum=self.S

        if(self.activationFunction==ActivationType.SIGMOID):
            self.a=self.sigmoid(sum)
            self.gradAF=self.a*(1-self.a)
        elif(self.activationFunction==ActivationType.TANH):
            self.a=self.tanh(sum)
            self.gradAF=(1-self.a*self.a)
        elif(self.activationFunction==ActivationType.RELU):
            self.a=self.Relu(sum)
            epsilon=1E-6
            self.gradAF = 1. * (self.a > epsilon)
            self.gradAF[self.gradAF == 0] = epsilon
            #self.gradAF=1.0*(self.a>0)
        elif(self.activationFunction==ActivationType.SOFTMAX):
            self.a=self.Softmax(sum)
            self.gradAF=None #we will use delta directly for back prop
        #apply dropOut unless last layer
        if(self.isLastLayer==False):
            self.mask = np.random.binomial(1, self.dropOut, size=self.a.shape)/self.dropOut
            self.a=self.mask*self.a
            self.gradAF=self.mask*self.gradAF
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def Tanh(self,x):
        return np.tanh(x)

    def Relu(self,x):
        der=np.copy(x)
        der[x<=0]=0
        return der
        #return np.maximum(x,0)

    def Softmax(self,x):
        ex = np.exp(x)
        if (x.shape[0] == x.size):
            return ex/ex.sum()
        for i in range(ex.shape[0]):
            denom = ex[i].sum()
            #print("ex ",ex[i])
            ex[i] = ex[i]/denom
        return ex



    



