class OneHotEncoder(object):
    def __init__(self, dataSet, featureIndex, featureName):
        self.featureName = featureName
        
        # get all the values of the feature that this encoder is encoding
        self.values = []
        for x in dataSet:
            if x[featureIndex] not in self.values:
                self.values.append(x[featureIndex])
                
        self.names = [ "is_%s_%s" % (self.featureName, value) for value in self.values ]            
        
        # little optimization
        self.valueToOneHotIDMap = {}
        for i in range(len(self.values)):
            self.valueToOneHotIDMap[self.values[i]] = i
        
        
    def GetOneHotEncodingFor(self, value):
        encoding = [ 0 for value in self.values ]
        
        try:
            encoding[self.valueToOneHotIDMap[value]] = 1
        except KeyError:
            # value wasn't present in the data used to create the encoding.
            #  so just leave all values as 0
            pass
        
        return encoding
    
    def GetFeatureCount(self):
        return len(self.values)
    
    def GetNameForIndex(self, index):
        return self.names[index]

class AdultFeaturize(object):
    """A class to convert the adult dataset into feature vectors that our learners can work with."""
    
    def __init__(self):
        # Hard code some information about the data set
        self.featureNames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        self.isNumeric     = [True ,False,True ,False,True ,False,False,False,False,False,True ,True ,True ,False]
        self.isCategorical = [False,True ,False,True ,False,True ,True ,True ,True ,True ,False,False,False,True ]

        # Leave out some features by default
        self.includeFeature    = [True ,True ,False,True, True ,True ,True ,True ,False,False,True ,True, True ,True ]
        
        self.featureSetCreated = False

    def CreateFeatureSet(self, xRaw, yRaw, useCategoricalFeatures=True, useNumericFeatures=False):     
        self.useFeature = [ self.includeFeature[i] and ((useCategoricalFeatures and self.isCategorical[i]) or (useNumericFeatures and self.isNumeric[i])) for i in range(len(self.featureNames)) ]
        
        self.oneHotEncoders = []
        for featureIndex in range(len(self.useFeature)):
            if self.useFeature[featureIndex] and self.isCategorical[featureIndex]:
                self.oneHotEncoders.append(OneHotEncoder(xRaw, featureIndex, self.featureNames[featureIndex]))
            else:
                self.oneHotEncoders.append(None)
                
        self.featureSetCreated = True

    def _FeaturizeX(self, xRaw):
        featureVector = []
        
        for i in range(len(self.useFeature)):
            if self.useFeature[i]:
                if self.isNumeric[i]:
                    featureVector.append(float(xRaw[i]))
                else:
                    featureVector = featureVector + self.oneHotEncoders[i].GetOneHotEncodingFor(xRaw[i])
                    
        return featureVector               

    def Featurize(self, xSetRaw):
        if not self.featureSetCreated:
            raise UserWarning("Trying to featurize before calling CreateFeatureSet")
        
        return [ self._FeaturizeX(x) for x in xSetRaw ]
        
    def GetFeatureCount(self):
        if not self.featureSetCreated:
            raise UserWarning("Trying to GetFeatureCount count before calling CreateFeatureSet")

        count = 0
        
        for i in range(len(self.useFeature)):
            if self.useFeature[i]:
                if self.isNumeric[i]:
                    count = count + 1
                else:
                    count = count + self.oneHotEncoders[i].GetFeatureCount()
                    
        return count
    
    def GetFeatureInfo(self, index):
        if not self.featureSetCreated:
            raise UserWarning("Trying to GetFeatureIndex before calling CreateFeatureSet")

        # this is in the space of the feature vector after one-hot encoding, so the index is 0 -> self.GetFeatureCount()
        currentIndex = 0
        
        # but we have to iterate over the raw features to find the index mapping
        for i in range(len(self.useFeature)):
            if self.useFeature[i]:
                if self.isNumeric[i]:
                    if currentIndex == index:
                        return self.featureNames[i]
                    else:
                        currentIndex = currentIndex + 1
                else: # is one hot encoded
                    nextIndex = currentIndex + self.oneHotEncoders[i].GetFeatureCount()
                    if index >= currentIndex and index < nextIndex:
                        return self.oneHotEncoders[i].GetNameForIndex(index - currentIndex)
                    else:
                        currentIndex = nextIndex

        raise UserWarning("Tried to get info for feature index %d but there are only %d features" % (index, self.GetFeatureCount()))