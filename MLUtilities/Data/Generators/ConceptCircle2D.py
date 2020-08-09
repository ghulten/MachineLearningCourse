import random
import math

def InCircle(x1, x2, radius):
    x = x1 - 0.5
    y = x2 - 0.5

    distance = math.sqrt(x*x + y*y)

    return 1 if distance < radius else 0

class ConceptCircle2D(object):
    def __init__(self, radius = 0.3):
        self.radius = radius
        
    def predict(self, x):
         y = [ InCircle(sample[0], sample[1], self.radius) for sample in x ]

         return y