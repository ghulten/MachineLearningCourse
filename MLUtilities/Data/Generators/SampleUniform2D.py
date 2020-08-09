import random
import datetime

class SampleUniform2D(object):
    def __init__(self, seed=None):
        if seed==None:
            random.seed(datetime.datetime.now())
        else:
            random.seed(seed)
            
        # used to produce the exace same sequence even if Generate is called multiple times
        self.nextSeed = random.random()
            
    def generate(self, numSamples):
        data = []

        for i in range(numSamples):
            random.seed(self.nextSeed)
            data.append([ random.random(), random.random()])
            self.nextSeed = random.random()

        return data
