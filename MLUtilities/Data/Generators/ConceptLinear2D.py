

import random
import MachineLearningCourse.MLUtilities.Learners.LinearHeuristicModel as LinearHeuristicModel

class ConceptLinear2D(object):
    def __init__(self, bias=0.0, weights=[1.0, 1.0]):
        self.model = LinearHeuristicModel.LinearHeuristicModel()
        self.model.fit([], [], bias, weights)
        
    def predict(self, x):
        return self.model.predict(x)