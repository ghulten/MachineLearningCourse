kOutputDirectory = "C:\\temp\\visualize"

import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamDataset as SMSSpamDataset

kDataPath = "MachineLearningCourse\\MLProjectSupport\\SMSSpam\\dataset\\SMSSpamCollection"

(xRaw, yRaw) = SMSSpamDataset.LoadRawData(kDataPath)

import MachineLearningCourse.MLUtilities.Data.Sample as Sample
(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

import MachineLearningCourse.MLUtilities.Learners.LogisticRegression as LogisticRegression
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.Assignments.Module01.SupportCode.SMSSpamFeaturize as SMSSpamFeaturize
import MachineLearningCourse.MLUtilities.Visualizations.Charting as Charting

# A helper function for calculating FN rate and FP rate across a range of thresholds
def TabulateModelPerformanceForROC(model, xValidate, yValidate):
   pointsToEvaluate = 100
   thresholds = [ x / float(pointsToEvaluate) for x in range(pointsToEvaluate + 1)]
   FPRs = []
   FNRs = []

   try:
      for threshold in thresholds:
         FPRs.append(EvaluateBinaryClassification.FalsePositiveRate(yValidate, model.predict(xValidate, classificationThreshold=threshold)))
         FNRs.append(EvaluateBinaryClassification.FalseNegativeRate(yValidate, model.predict(xValidate, classificationThreshold=threshold)))
   except NotImplementedError:
      raise UserWarning("The 'model' parameter must have a 'predict' method that supports using a 'classificationThreshold' parameter with range [ 0 - 1.0 ] to create classifications.")

   return (FPRs, FNRs, thresholds)


# Hyperparameters to use for the run
stepSize = 0.1
convergence = 0.0001

# Set up to hold information for creating ROC curves
seriesFPRs = []
seriesFNRs = []
seriesLabels = []

#### Learn a model with 25 frequent features
featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
featurizer.CreateVocabulary(xTrainRaw, yTrain, numFrequentWords = 25)

xTrain      = featurizer.Featurize(xTrainRaw)
xValidate   = featurizer.Featurize(xValidateRaw)
xTest       = featurizer.Featurize(xTestRaw)

model = LogisticRegression.LogisticRegression()
model.fit(xTrain,yTrain,convergence=convergence, stepSize=stepSize)

(modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xValidate, yValidate)
seriesFPRs.append(modelFPRs)
seriesFNRs.append(modelFNRs)
seriesLabels.append('25 Frequent')

#### Learn a model with 25 features by mutual information
featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=False)
featurizer.CreateVocabulary(xTrainRaw, yTrain, numMutualInformationWords = 25)

xTrain      = featurizer.Featurize(xTrainRaw)
xValidate   = featurizer.Featurize(xValidateRaw)
xTest       = featurizer.Featurize(xTestRaw)

model = LogisticRegression.LogisticRegression()
model.fit(xTrain,yTrain,convergence=convergence, stepSize=stepSize)

(modelFPRs, modelFNRs, thresholds) = TabulateModelPerformanceForROC(model, xValidate, yValidate)
seriesFPRs.append(modelFPRs)
seriesFNRs.append(modelFNRs)
seriesLabels.append('25 Mutual Information')

Charting.PlotROCs(seriesFPRs, seriesFNRs, seriesLabels, useLines=True, chartTitle="ROC Comparison", xAxisTitle="False Negative Rate", yAxisTitle="False Positive Rate", outputDirectory=kOutputDirectory, fileName="Plot-SMSSpamROCs")