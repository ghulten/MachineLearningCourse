import time
import math
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryProbabilityEstimate as EvaluateBinaryProbabilityEstimate


class LogisticRegression(object):
    """Stub class for a Logistic Regression Model"""

    def __init__(self, featureCount=None):
        self.isInitialized = False
        
        if featureCount != None:
            self.__initialize(featureCount)

    def __testInput(self, x, y):
        if len(x) == 0:
            raise UserWarning("Trying to fit but can't fit on 0 training samples.")

        if len(x) != len(y):
            raise UserWarning("Trying to fit but length of x != length of y.")

    def __initialize(self, featureCount):
        self.weights = [ 0.0 for i in range(featureCount) ]
        self.weight0 = 0.0
        
        self.converged = False
        self.totalGradientDescentSteps = 0

        self.isInitialized = True

    def loss(self, x, y):
        return EvaluateBinaryProbabilityEstimate.LogLoss(y, self.predictProbabilities(x))

    def predictProbabilities(self, x):
        # For each sample do the dot product between features and weights (remember the bias weight, weight0)
        #  pass the results through the sigmoid function to convert to probabilities.
        
        print("Stub predictProbabilities in ", __file__)
        
    def predict(self, x, classificationThreshold = 0.5):
        print("Stub predict in ", __file__)
        
    def __gradientDescentStep(self, x, y, stepSize):
        self.totalGradientDescentSteps = self.totalGradientDescentSteps + 1
        
        print("Stub gradientDescentStep in ", __file__)

    # Allows you to partially fit, then pause to gather statistics / output intermediate information, then continue fitting
    def incrementalFit(self, x, y, maxSteps=1, stepSize=1.0, convergence=0.005):
        self.__testInput(x,y)
        if self.isInitialized == False:
            self.__initialize(len(x[0]))
        
        # do a maximum of 'maxSteps' of gradient descent with the indicated stepSize (use the __gradientDescentStep stub function for code clarity).
        #  stop and set self.converged to true if the mean log loss on the training set decreases by less than 'convergence' on a gradient descent step.
        
        print("Stub incrementalFit in ", __file__)

    def fit(self, x, y, maxSteps=50000, stepSize=1.0, convergence=0.005, verbose = True):
        
        startTime = time.time()
        
        self.incrementalFit(x,y,maxSteps=maxSteps, stepSize=stepSize, convergence=convergence)
        
        endTime = time.time()
        runtime = endTime - startTime
      
        if not self.converged:
            print("Warning: did not converge after taking the maximum allowed number of steps.")
        elif verbose:
            print("LogisticRegression converged in %d steps (%.2f seconds) -- %d features. Hyperparameters: stepSize=%f and convergence=%f." % (self.totalGradientDescentSteps, runtime, len(self.weights), stepSize, convergence))

    def visualize(self):
        print("w0: %f " % (self.weight0), end='')

        for i in range(len(self.weights)):
            print("w%d: %f " % (i+1, self.weights[i]), end='')

        print("\n")

