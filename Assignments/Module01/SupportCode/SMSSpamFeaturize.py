class SMSSpamFeaturize(object):
    """A class to coordinate turning SMS spam strings into feature vectors."""

    def __init__(self, useHandCraftedFeatures=False):
        # use hand crafted features specified in _FeaturizeXForHandCrafted()
        self.useHandCraftedFeatures = useHandCraftedFeatures
        
        self.ResetVocabulary()
        
    def ResetVocabulary(self):
        self.vocabularyCreated = False
        self.vocabulary = []

    def Tokenize(self, xRaw):
        return str.split(xRaw)
    
    def FindMostFrequentWords(self, x, n):
        print("Stub FindMostFrequentWords in ", __file__)

    def FindTopWordsByMutualInformation(self, x, y, n):
        print("Stub FindTopWordsByMutualInformation in ", __file__)

    def CreateVocabulary(self, xTrainRaw, yTrainRaw, supplementalVocabularyWords=[]):
        if self.vocabularyCreated:
            raise UserWarning("Calling CreateVocabulary after the vocabulary was already created. Call ResetVocabulary to reinitialize.")
            
        # This function will eventually scan the strings in xTrain and choose which words to include in the vocabulary.
        #   But don't implement that until you reach the assignment that requires it...
        
        # For now, only use words that are passed in
        self.vocabulary = self.vocabulary + supplementalVocabularyWords
        
        self.vocabularyCreated = True
        
    def _FeaturizeXForVocabulary(self, xRaw): 
        features = []
        
        # for each word in the vocabulary output a 1 if it appears in the SMS string, or a 0 if it does not
        tokens = self.Tokenize(xRaw)
        for word in self.vocabulary:
            if word in tokens:
                features.append(1)
            else:
                features.append(0)
                
        return features

    def _FeaturizeXForHandCraftedFeatures(self, xRaw):
        features = []
        
        # This function can produce an array of hand-crafted features to add on top of the vocabulary related features
        if self.useHandCraftedFeatures:
            # Have a feature for longer texts
            if(len(xRaw)>40):
                features.append(1)
            else:
                features.append(0)

            # Have a feature for texts with numbers in them
            if(any(i.isdigit() for i in xRaw)):
                features.append(1)
            else:
                features.append(0)
            
        return features

    def _FeatureizeX(self, xRaw):
        return self._FeaturizeXForVocabulary(xRaw) + self._FeaturizeXForHandCraftedFeatures(xRaw)

    def Featurize(self, xSetRaw):
        return [ self._FeatureizeX(x) for x in xSetRaw ]

    def GetFeatureInfo(self, index):
        if index < len(self.vocabulary):
            return self.vocabulary[index]
        else:
            # return the zero based index of the heuristic feature
            return "Heuristic_%d" % (index - len(self.vocabulary))
