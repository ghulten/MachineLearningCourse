import MachineLearningCourse.MLProjectSupport.Blink.BlinkDataset as BlinkDataset

(xRaw, yRaw) = BlinkDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)

print("Train is %d samples, %.4f percent opened." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent opened." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent opened" % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

from PIL import Image
import torchvision.transforms as transforms
import torch 

##
# Load the images, normalize and convert them into tensors
##

transform = transforms.Compose([
            transforms.ToTensor()
            ,transforms.Normalize(mean=[0.], std=[0.5])
            ])

xTrainImages = [ Image.open(path) for path in xTrainRaw ]
xTrain = torch.stack([ transform(image) for image in xTrainImages ])

yTrain = torch.Tensor([ [ yValue ] for yValue in yTrain ])

xValidateImages = [ Image.open(path) for path in xValidateRaw ]
xValidate = torch.stack([ transform(image) for image in xValidateImages ])

yValidate = torch.Tensor([ [ yValue ] for yValue in yValidate ])

xTestImages = [ Image.open(path) for path in xTestRaw ]
xTest = torch.stack([ transform(image) for image in xTestImages ])

yTest = torch.Tensor([ [ yValue ] for yValue in yTest ])


######
######

import MachineLearningCourse.Assignments.Module03.SupportCode.BlinkNeuralNetwork as BlinkNeuralNetwork

# Create the model
model = BlinkNeuralNetwork.BlinkNeuralNetwork(hiddenNodes = 5)

# Create the loss function to use (Mean Square Error)
lossFunction = torch.nn.MSELoss(reduction='sum')

# Create the optimization method (Stochastic Gradient Descent) and the step size (lr -> learning rate)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

##
# Move the model and data to the GPU if you're using your GPU
##

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is:", device)

model.to(device)

# Move the data onto whichever device was selected
xTrain = xTrain.to(device)
yTrain = yTrain.to(device)
xValidate = xValidate.to(device)
yValidate = yValidate.to(device)
xTest = xTest.to(device)
yTest = yTest.to(device)

##
# Build the model
##
import time
startTime = time.time()

trainLosses = []
validationLosses = []

converged = False
epoch = 1
lastValidationLoss = None

while not converged and epoch < 1000:
    # Do the forward pass
    yTrainPredicted = model(xTrain)

    # Compute the total loss summed across training samples in the epoch
    #  note this is different from our implementation, which took one step
    #  of gradient descent per sample.
    trainLoss = lossFunction(yTrainPredicted, yTrain)
    
    # Reset the gradients in the network to zero
    optimizer.zero_grad()

    # Backprop the errors from the loss on this iteration
    trainLoss.backward()

    # Do a weight update step
    optimizer.step()
    
    # now check the validation loss
    model.train(mode=False)
    yValidatePredicted = model(xValidate)
    validationLoss = lossFunction(yValidatePredicted, yValidate)

    trainLosses.append(trainLoss.item() / len(xTrain))
    validationLosses.append(validationLoss.item() / len(xValidate))
    
    if lastValidationLoss != None and validationLoss.item() > lastValidationLoss:
        converged = True
        pass
    else:
        lastValidationLoss = validationLoss.item()
        
    epoch = epoch + 1

    model.train(mode=True)

endTime = time.time()

print("Runtime: %s" % (endTime - startTime))

##
# Visualize Training run
##

kOutputDirectory = "C:\\temp\\visualize"

import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

xValues   = [ i + 1 for i in range(len(trainLosses))]

Charting.PlotSeries([trainLosses, validationLosses], ["Train Loss", "Validate Loss"], xValues, useMarkers=False, chartTitle="Pytorch First Modeling Run", xAxisTitle="Epoch", yAxisTitle="Loss", yBotLimit=0.0, outputDirectory=kOutputDirectory, fileName="PyTorch-Initial-TrainValidate")


##
# Evaluate the Model
##

import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds

model.train(mode=False)
yTestPredicted = model(xTest)

testAccuracy = EvaluateBinaryClassification.Accuracy(yTest, [ 1 if pred > 0.5 else 0 for pred in yTestPredicted ])
print("Accuracy simple:", testAccuracy, ErrorBounds.GetAccuracyBounds(testAccuracy, len(yTestPredicted), 0.95))


