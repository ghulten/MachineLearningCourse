import random
import math

import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryProbabilityEstimate as EvaluateBinaryProbabilityEstimate

class NeuralNetworkFullyConnected(object):
    """Framework for fully connected neural network"""
    def __init__(self, numInputFeatures, hiddenLayersNodeCounts=[2], seed=1000):
        random.seed(seed)

        self.totalEpochs = 0
        self.lastLoss    = None
        self.converged   = False

        # set up the input layer
        self.layerWidths = [ numInputFeatures ]

        # set up the hidden layers
        for i in range(len(hiddenLayersNodeCounts)):
            self.layerWidths.append(hiddenLayersNodeCounts[i])

        # output layer
        self.layerWidths.append(1)

        ###
        ## now set up all the parameters and any arrays you want for forward/backpropagation
        ###
        
        print("Stub __init__ in ", __file__)
            

    def feedForward(self, x):     
        print("Stub feedForward in ", __file__)

    def backpropagate(self, y):
        print("Stub backpropogate in ", __file__)

    def updateweights(self, step, momentum):
        print("Stub updateweights in ", __file__)

    def loss(self, x, y):        
        return EvaluateBinaryProbabilityEstimate.LogLoss(y, self.predictProbabilities(x))

    def predictOneProbability(self, x):
        self.feedForward(x)

        print("Stub predictOneProbability in ", __file__)
        
        # return the activation of the neuron in the output layer
        return -1

    def predictProbabilities(self, x):    
        return [ self.predictOneProbability(sample) for sample in x ]

    def predict(self, x, threshold = 0.5):
        return [ 1 if probability > threshold else 0 for probability in self.predictProbabilities(x) ]
    
    def __CheckForConvergence(self, x, y, convergence):
        loss = self.loss(x,y)

        if self.lastLoss != None:
            deltaLoss = abs(self.lastLoss - loss)
            self.converged = deltaLoss < convergence
            
        self.lastLoss = loss
    
    # Allows you to partially fit, then pause to gather statistics / output intermediate information, then continue fitting
    def incrementalFit(self, x, y, epochs=1, step=0.01, convergence = None):
        for _ in range(epochs):
            if self.converged:
                return
        
            # do a full epoch of stocastic gradient descent
            print("Stub incrementalFit in ", __file__)

            if convergence != None:
                self.__CheckForConvergence(x, y, convergence)
             
                
    def fit(self, x, y, maxEpochs=50000, stepSize=0.01, convergence=0.00001, verbose = True):        
        startTime = time.time()
        
        self.incrementalFit(x, y, maxEpochs=maxEpochs, stepSize=stepSize, convergence=convergence)
        
        endTime = time.time()
        runtime = endTime - startTime
      
        if not self.converged:
            print("Warning: NeuralNetwork did not converge after the maximum allowed number of epochs.")
        elif verbose:
            print("NeuralNetwork converged in %d epochs (%.2f seconds) -- %d features. Hyperparameters: stepSize=%f and convergence=%f." % (self.totalEpochs, runtime, len(x[0]), stepSize, convergence))

