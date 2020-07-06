import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamSupport as SMSSpamSupport

kDataPath = "MachineLearningCourse\\MLProjectSupport\\SMSSpam\\dataset\\SMSSpamCollection"

(xRaw, yRaw) = SMSSpamSupport.LoadRawData(kDataPath)
(xTrainRaw, yTrain, xValidateRaw, yValidate, xTestRaw, yTest) = SMSSpamSupport.TrainValidateTestSplit(xRaw, yRaw, percentValidate=.1, percentTest=.1)

print("Statistics on the data sets:")
print(" Train set contains %04d samples,    percent spam: " % (len(yTrain))    + "{:.2%}".format(sum(yTrain)/len(yTrain)))
print(" Validate set contains %04d samples, percent spam: " % (len(yValidate)) + "{:.2%}".format(sum(yValidate)/len(yValidate)))
print(" Test set contains %04d samples,     percent spam: " % (len(yTest))     + "{:.2%}".format(sum(yTest)/len(yTest)))


# This time we aren't going to use the xRaw values - we are going to convert our xRaws into feature vectors
import MachineLearningCourse.MLProjectSupport.SMSSpam.SMSSpamFeaturize as SMSSpamFeaturize

# Create an instance of the featurizer, and tell it to use some hand-crafted code we created to produce features.
featurizer = SMSSpamFeaturize.SMSSpamFeaturize(useHandCraftedFeatures=True)

# We'll also create a vocabulary and use the presence or abscence of specific words in the feature vector.

#  In a later assignment, you'll update 'CreateVocabulary' to select the vocabulary automatically. For now, just add in a few 'spammy' words by hand.
featurizer.CreateVocabulary(xTrainRaw, yTrain, supplementalVocabularyWords=['call','to','your'])

# Apply the featurerizer to the raw data sets to produce feature vectors.
xTrain      = featurizer.Featurize(xTrainRaw)
xValidate   = featurizer.Featurize(xValidateRaw)
xTest       = featurizer.Featurize(xTestRaw)

print("\n - Inspect the features -")
for i in range(len(xTrain[0])):
    print(featurizer.GetFeatureInfo(i))


print("\n - Inspect feature values for a few training samples -")
for i in range(5):
    print(yTrain[i], "-", xTrain[i], xTrainRaw[i])
    
# Now let's up our modeling game (as compared to predicting the most common class)
#  we'll use a heuristic (hand-tuned) linear model.
import MachineLearningCourse.MLUtilities.Learners.LinearHeuristicModel as LinearHeuristicModel
model = LinearHeuristicModel.LinearHeuristicModel()

model.fit(xTrain, yTrain, -1.0, [ .75, .75, .75, .25, .25 ])

print("\n - Inspect the weights on the heuristically-tuned model -")
model.visualize()

yValidatePredicted = model.predict(xValidate)
    
import MachineLearningCourse.MLUtilities.Evaluations.EvaluateBinaryClassification as EvaluateBinaryClassification
EvaluateBinaryClassification.ExecuteAll(yValidate, yValidatePredicted)
