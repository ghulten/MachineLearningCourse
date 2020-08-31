kOutputDirectory = "C:\\temp\\visualize"

import MachineLearningCourse.MLProjectSupport.Blink.BlinkDataset as BlinkDataset

### UPDATE this path for your environment
kDataPath = "MachineLearningCourse\\MLProjectSupport\\Blink\\dataset\\"

(xRaw, yRaw) = BlinkDataset.LoadRawData(kDataPath)

import MachineLearningCourse.MLUtilities.Data.Sample as Sample

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)

print("Train is %d samples, %.4f percent opened." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent opened." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent opened" % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

import MachineLearningCourse.Assignments.Module03.SupportCode.BlinkFeaturize as BlinkFeaturize

featurizer = BlinkFeaturize.BlinkFeaturize()

sampleStride = 2
featurizer.CreateFeatureSet(xTrainRaw, yTrain, includeEdgeFeatures=False, includeIntensities=True, intensitiesSampleStride=sampleStride)

xTrain    = featurizer.Featurize(xTrainRaw)
xValidate = featurizer.Featurize(xValidateRaw)
xTest     = featurizer.Featurize(xTestRaw)

import MachineLearningCourse.MLUtilities.Learners.NeuralNetworkFullyConnected as NeuralNetworkFullyConnected
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

import time

from joblib import Parallel, delayed

import PIL
from PIL import Image

def VisualizeWeights(weightArray, outputPath, sampleStride = 2):
    imageDimension = int(24 / sampleStride)
    pixelSize = 2 * sampleStride
    imageSize = imageDimension * pixelSize

    # note the extra weight for the bias is where the +1 comes from
    if len(weightArray) != (imageDimension * imageDimension) + 1:
        raise UserWarning("size of the weight array is %d but it should be %d" % (len(weightArray), (imageDimension * imageDimension) + 1))

    if not outputPath.endswith(".jpg"):
        raise UserWarning("output path should be a path to a file that ends in .jpg, it is currently: %s" % (outputPath))

    image = Image.new("RGB", (imageSize, imageSize), "White")

    pixels = image.load()

    for x in range(imageDimension):
        for y in range(imageDimension):
            weight = weightArray[1+(x*imageDimension) + y]
            
            # Add in the bias to help understand the weight's function
            weight += weightArray[0]
            
            if weight >= 0:
                color = (0, int(255 * abs(weight)), 0)
            else:
                color = (int(255 * abs(weight)), 0, 0)
            
            for i in range(pixelSize):
                for j in range(pixelSize):
                    pixels[(x * pixelSize) + i, (y * pixelSize) + j] = color

    image.save(outputPath)

hiddenStructure = [ 2 ]
    
model = NeuralNetworkFullyConnected.NeuralNetworkFullyConnected(len(xTrain[0]), hiddenLayersNodeCounts=hiddenStructure)

for filterNumber in range(hiddenStructure[0]):
    ## update the first parameter based on your representation
    VisualizeWeights(model.layers[1][filterNumber], "%s\\filters\\epoch%d_neuron%d.jpg" % (kOutputDirectory, 0, filterNumber), sampleStride=sampleStride)

maxEpochs = 1000
step = 1.0
convergence = 0.1

for i in range(maxEpochs): 
    if not model.converged:
        model.incrementalFit(xTrain, yTrain, epochs = 1, step=step, convergence=convergence)
        
        for filterNumber in range(hiddenStructure[0]):
            ## update the first parameter based on your representation
            VisualizeWeights(model.layers[1][filterNumber], "%s\\filters\\epoch%d_neuron%d.jpg" % (kOutputDirectory, i+1, filterNumber), sampleStride=sampleStride)

# Evaluate things...