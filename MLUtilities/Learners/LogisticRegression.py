import math
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryProbabilityEstimate as EvaluateBinaryProbabilityEstimate


class LogisticRegression(object):
    """Stub class for a Logistic Regression Model"""

    def __init__(self, featureCount=None):
        self.weights = None
        self.weight0 = None
        self.converged = False
        
        if featureCount != None:
            self.__initializeWeights(featureCount)

    def __testInput(self,x,y):
        if len(x) == 0:
            raise UserWarning("Trying to fit but can't fit on 0 training samples.")

        if len(x) != len(y):
            raise UserWarning("Trying to fit but length of x != length of y.")

    def __initializeWeights(self, featureCount):
        self.weights = [ 0.0 for i in range(featureCount) ]
        self.weight0 = 0.0

    def __gradientDescentStep(self, x, y, stepSize):
        print("Stub gradientSescentStep in ", __file__)

    def fit(self, x, y, stepSize=0.01, iterations=1000, convergence=0.01, verbose=False):
        self.__testInput(x,y)
        if self.weight0 == None:
            self.__initializeWeights(len(x[0]))
        
        if self.converged == False:
            pass
            # do a maximum of 'iterations' steps of gradient descent with the indicated stepSize.
            #  converge if the mean log loss on the training set decreases by less than 'convergence' on a gradient descent step.
            #  use 'verbose' to output debugging information if you want.
        
        print("Stub fit in ", __file__)

    def loss(self, x, y):
        return EvaluateBinaryProbabilityEstimate.LogLoss(y, self.predictProbabilities(x))

    def predictProbabilities(self, x):
        # For each sample do the dot product with the weights (remember bias)
        #  pass the result through a sigmoid function to convert to a probability.
        
        print("Stub predictProbabilities in ", __file__)
        
    
    def predict(self, x, classificationThreshold = 0.5):
        print("Stub predict in ", __file__)


    def visualize(self):
        print("w0: %f " % (self.weight0), end='')

        for i in range(len(self.weights)):
            print("w%d: %f " % (i+1, self.weights[i]), end='')

        print("\n")

