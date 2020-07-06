
import SMSSpamSupport
import EvaluateBinaryClassification

### UPDATE this path for your environment
kDataPath = "MachineLearningCourse\\MLProjectSupport\\SMSSpam\\dataset\\SMSSpamCollection"

(xRaw, yRaw) = SMSSpamSupport.LoadRawData(kDataPath)

(xTrainRaw, yTrainRaw, xValidateRaw, yValidateRaw, xTestRaw, yTestRaw) = SMSSpamSupport.TrainValidateTestSplit(xRaw, yRaw)

print("Train set contains %04d samples,    percent spam: " % (len(yTrainRaw))    + "{:.2%}".format(sum(yTrainRaw)/len(yTrainRaw)))
print("Validate set contains %04d samples, percent spam: " % (len(yValidateRaw)) + "{:.2%}".format(sum(yValidateRaw)/len(yValidateRaw)))
print("Test set contains %04d samples,     percent spam: " % (len(yTestRaw))     + "{:.2%}".format(sum(yTestRaw)/len(yTestRaw)))


(xTrain, xValidate, xTest) = SMSSpamSupport.Featurize(xTrainRaw, xValidateRaw, xTestRaw)
yTrain = yTrainRaw
yValidate = yValidateRaw
yTest = yTestRaw

############################
import MostCommonModel

model = MostCommonModel.MostCommonModel()
model.fit(xTrain, yTrain)
yValidatePredicted = model.predict(xValidate)

print("### 'Most Common' model")

EvaluateBinaryClassification.ExecuteAll(yValidate, yValidatePredicted)

############################
import SpamHeuristicModel
model = SpamHeuristicModel.SpamHeuristicModel()
model.fit(xTrain, yTrain)
yValidatePredicted = model.predict(xValidate)

print("### Heuristic model")

EvaluateBinaryClassification.ExecuteAll(yValidate, yValidatePredicted)

############################
# import LogisticRegressionModel
# model = LogisticRegressionModel.LogisticRegressionModel()

# print("Logistic regression model")
# for i in [50000]:
#     model.fit(xTrain, yTrain, iterations=i, step=0.01)
#     yValidatePredicted = model.predict(xValidate)
    
#     print("%d, %f, %f, %f" % (i, model.weights[1], model.loss(xValidate, yValidate), EvaluateBinaryClassification.Accuracy(yValidate, yValidatePredicted)))

# EvaluateBinaryClassification.ExecuteAll(yValidate, yValidatePredicted)

# Don't use the test data yet...save that for after we do some more serious feature engineering.
