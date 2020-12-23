kOutputDirectory = "C:\\temp\\visualize"

import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

(xRaw, yRaw) = SMSSpamDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Data.CrossValidation as CrossValidation

import time

# A helper function for calculating FN rate and FP rate across a range of thresholds
def TabulateModelPerformanceForROC(model, xValidate, yValidate):
   pointsToEvaluate = 100
   thresholds = [ x / float(pointsToEvaluate) for x in range(pointsToEvaluate + 1)]
   FPRs = []
   FNRs = []
   yPredicted = model.predictProbabilities(xValidate)

   try:
      for threshold in thresholds:
        yHats = [ 1 if pred > threshold else 0 for pred in yPredicted ]
        FPRs.append(EvaluateBinaryClassification.FalsePositiveRate(yValidate, yHats))
        FNRs.append(EvaluateBinaryClassification.FalseNegativeRate(yValidate, yHats))
   except NotImplementedError:
      raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

   return (FPRs, FNRs, thresholds)

import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting
## This function will help you plot with error bars. Use it just like PlotSeries, but with parallel arrays of error bar sizes in the second variable
#     note that the error bar size is drawn above and below the series value. So if the series value is .8 and the confidence interval is .78 - .82, then the value to use for the error bar is .02

# Charting.PlotSeriesWithErrorBars([series1, series2], [errorBarsForSeries1, errorBarsForSeries2], ["Series1", "Series2"], xValues, chartTitle="<>", xAxisTitle="<>", yAxisTitle="<>", yBotLimit=0.8, outputDirectory=kOutputDirectory, fileName="<name>")


## This helper function should execute a single run and save the results on 'runSpecification' (which could be a dictionary for convienience)
#    for later tabulation and charting...
def ExecuteEvaluationRun(runSpecification, xTrainRaw, yTrain, numberOfFolds = 5):
    startTime = time.time()
    
    # HERE upgrade this to use crossvalidation
    
    featurizer = SMSSpamFeaturize.SMSSpamFeaturize()
    featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = runSpecification['numFrequentWords'], numMutualInformationWords = runSpecification['numMutualInformationWords'])

    xTrain      = featurizer.Featurize(xTrainRaw)
    xValidate   = featurizer.Featurize(xValidateRaw)

    model = LogisticRegression.LogisticRegression()
    model.fit(xTrain,yTrain,convergence=runSpecification['convergence'], stepSize=runSpecification['stepSize'], verbose=True)
    
    validationSetAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, model.predict(xValidate))
    
    runSpecification['accuracy'] = validationSetAccuracy

    # HERE upgrade this to calculate and save some error bounds...
    
    endTime = time.time()
    runSpecification['runtime'] = endTime - startTime
    
    return runSpecification

evaluationRunSpecifications = []
for numMutualInformationWords in [10, 25, 50, 75, 100]:

    runSpecification = {}
    runSpecification['optimizing'] = 'numMutualInformationWords'
    runSpecification['numMutualInformationWords'] = numMutualInformationWords
    runSpecification['stepSize'] = 1.0
    runSpecification['convergence'] = 0.005
    runSpecification['numFrequentWords'] = 0
    
    evaluationRunSpecifications.append(runSpecification)

## if you want to run in parallel you need to install joblib as described in the lecture notes and adjust the comments on the next three lines...
#from joblib import Parallel, delayed
#evaluations = Parallel(n_jobs=12)(delayed(ExecuteEvaluationRun)(runSpec, xTrainRaw, yTrain) for runSpec in evaluationRunSpecifications)

evaluations = [ ExecuteEvaluationRun(runSpec, xTrainRaw, yTrain) for runSpec in evaluationRunSpecifications ]

for evaluation in evaluations:
    print(evaluation)
    
# Good luck!