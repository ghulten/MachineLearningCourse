import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamSupport as SMSSpamSupport

### UPDATE this path for your environment
kDataPath = "MachineLearningCourse\\MLProjectSupport\\SMSSpam\\dataset\\SMSSpamCollection"

# x represents training data, y represents the labels. These are parallel arrays.
#  'Raw' indicates that the data has not been processed into features.
#    in this case, the xRaw array contains the raw SMS text strings and yRaw contains 1 if the message is spam and 0 if it isn't.
(xRaw, yRaw) = SMSSpamSupport.LoadRawData(kDataPath)

# The 'TrainValidateTestSplit' function separates the raw data into three sets to use for your modeling process. These are:
#  1) the training data, which you should use to build your model and make any feature creation decision
#  2) the validation data, which you should use to tune your modeling process (hyper-parameters, etc)
#  3) the testing data, which you should use sparingly to estimate the quality of your final model
#
# In this case, use 80% of data for training, 10% for validation, and 10% for testing.
(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = SMSSpamSupport.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

# Now do some basic data exploration. Always a good idea to look at some very basic stats about your data sets 
print("Statistics on the data sets:")
print(" Train set contains %04d samples,    percent spam: " % (len(yTrain))    + "{:.2%}".format(sum(yTrain)/len(yTrain)))
print(" Validate set contains %04d samples, percent spam: " % (len(yValidate)) + "{:.2%}".format(sum(yValidate)/len(yValidate)))
print(" Test set contains %04d samples,     percent spam: " % (len(yTest))     + "{:.2%}".format(sum(yTest)/len(yTest)))

# and to examing a few elements of the training set
print("\n- Inspect a few training samples -")
for i in range(5):
    print(" %d - %s" % (yTrain[i], xTrainRaw[i]))


# Now we'll do our first 'machine learning' using a very simple model.
import MachineLearningCourse.MLUtilities.Learners.MostCommonClassModel as MostCommonClassModel

model = MostCommonClassModel.MostCommonClassModel()

# go read the ModelMostCommon code to see what model.fit does
model.fit(xTrainRaw, yTrain)

print("\n- Inspect the model -")
model.visualize()

# model.predict takes the 'features' (in this case the raw strings) and returns a parallel array of preditions (in this case the most common label in the training set).
yTrainPredicted = model.predict(xTrainRaw)

# look at a few of the predictions, along with the correct labels and the raw x data.
print("\n- Inspect a few predictions [ <predicted> (<true label>) - <raw string> ] -")
for i in range(5):
    print("%d (%d) - %s" % (yTrainPredicted[i], yTrain[i], xTrainRaw[i]))

import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
trainSetAccuracy = EvaluateBinaryClassification.Accuracy(yTrain, yTrainPredicted)

print("\n---")
print("Predicting the most common class gives: {:.2%} accuracy on the training set.".format(trainSetAccuracy))

# Now evaluate the ModelMostCommon that we just fit on the training data on the validation set and output the accuracy