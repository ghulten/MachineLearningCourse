import MachineLearningCourse.MLUtilities.Image.Convolution2D as Convolution2D
from PIL import Image

import time
from joblib import Parallel, delayed

class BlinkFeaturize(object):
    def __init__(self):
        self.featureSetCreated = False
    
    def CreateFeatureSet(self, xRaw, yRaw, includeEdgeFeatures=True, includeRawPixels=False, includeIntensities=False, intensitiesSampleStride = 2):
        self.includeEdgeFeatures = includeEdgeFeatures
        self.includeRawPixels = includeRawPixels
        self.includeIntensities = includeIntensities
        self.intensitiesSampleStride = int(intensitiesSampleStride)
        
        self.featureSetCreated = True
        
    def _FeaturizeX(self, xRaw):
        featureVector = []
        
        image = Image.open(xRaw)
        
        xSize = image.size[0]
        ySize = image.size[1]
        numPixels = xSize * ySize

        pixels = image.load()

        if self.includeEdgeFeatures:
            yEdges = Convolution2D.Convolution3x3(image, Convolution2D.SobelY)
            xEdges = Convolution2D.Convolution3x3(image, Convolution2D.SobelX)
            
            avgYEdge = sum([sum([abs(value) for value in row]) for row in yEdges]) / numPixels
            avgXEdge = sum([sum([abs(value) for value in row]) for row in xEdges]) / numPixels
            
            featureVector.append(avgYEdge)
            featureVector.append(avgXEdge)


        if self.includeRawPixels:
            for x in range(xSize):
                for y in range(ySize):
                    featureVector.append(pixels[x,y])

        if self.includeIntensities:
            for x in range(0, xSize, self.intensitiesSampleStride):
                for y in range(0, ySize, self.intensitiesSampleStride):
                    featureVector.append(pixels[x,y]/255.0)

        return featureVector

    def Featurize(self, xSetRaw, verbose = True):
        if not self.featureSetCreated:
            raise UserWarning("Trying to featurize before calling CreateFeatureSet")
        
        if verbose:
            print("Loading & featurizing %d image files..." % (len(xSetRaw)))
        
        
        startTime = time.time()

        # If you don't have joblib installed you can swap these comments
        # result = [ self._FeaturizeX(x) for x in xSetRaw ]
        
        result = Parallel(n_jobs=12)(delayed(self._FeaturizeX)(x) for x in xSetRaw)
    
        endTime = time.time()
        runtime = endTime - startTime
        
        if verbose:
            print("   Complete in %.2f seconds" % (runtime))
        
        return result
