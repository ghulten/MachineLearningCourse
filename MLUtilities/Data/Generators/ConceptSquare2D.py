import random

class ConceptSquare2D(object):
    def __init__(self, width=.3, seed=None):
        if seed != None:
            random.seed(seed)
            
        self.width = width
        self.center = [ random.random(), random.random() ]
        
    def predict(self, x):
        return [ 1 if (abs(sample[0]-self.center[0]) <= self.width ) and ( abs(sample[1]-self.center[1]) <= self.width ) else 0 for sample in x ]
    