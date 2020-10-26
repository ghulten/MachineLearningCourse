kOutputDirectory = "C:\\temp\\visualize"

import MachineLearningCourse.MLProjectSupport.Adult.AdultDataset as AdultDataset

(xRaw, yRaw) = AdultDataset.LoadRawData()

import MachineLearningCourse.MLUtilities.Data.Sample as Sample

(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = Sample.TrainValidateTestSplit(xRaw, yRaw)

print("Train is %d samples, %.4f percent >50K." % (len(yTrain), 100.0 * sum(yTrain)/len(yTrain)))
print("Validate is %d samples, %.4f percent >50K." % (len(yValidate), 100.0 * sum(yValidate)/len(yValidate)))
print("Test is %d samples %.4f percent >50K." % (len(yTest), 100.0 * sum(yTest)/len(yTest)))

import MachineLearningCourse.Assignments.Module02.SupportCode.AdultFeaturize as AdultFeaturize

featurizer = AdultFeaturize.AdultFeaturize()
featurizer.CreateFeatureSet(xTrainRaw, yTrain, useCategoricalFeatures = True, useNumericFeatures = False)

for i in range(featurizer.GetFeatureCount()):
    print("%d - %s" % (i, featurizer.GetFeatureInfo(i)))

xTrain    = featurizer.Featurize(xTrainRaw)
xValidate = featurizer.Featurize(xValidateRaw)
xTest     = featurizer.Featurize(xTestRaw)

for i in range(10):
    print("%d - " % (yTrain[i]), xTrain[i])

############################
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
import MachineLearningCourse.MLUtilities.Evaluations.ErrorBounds as ErrorBounds
import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel

model = MostCommonClassModel.MostCommonClassModel()
model.fit(xTrain, yTrain)
yValidatePredicted = model.predict(xValidate)
validateAccuracy = EvaluateBinaryClassification.Accuracy(yValidate, yValidatePredicted)
errorBounds = ErrorBounds.GetAccuracyBounds(validateAccuracy, len(yValidate), 0.95)

print()
print("### 'Most Common Class' model validate set accuracy: %.4f (95%% %.4f - %.4f)" % (validateAccuracy, errorBounds[0], errorBounds[1]))
