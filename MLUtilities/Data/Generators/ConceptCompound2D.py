class ConceptCompound2D(object):

    def __init__(self, concepts=[]):
        self.concepts = concepts

    def _predictSingle(self, sample):
        return 1 if sum([ concept.predict( [ sample ] )[0] for concept in self.concepts ]) > 0 else 0

    def predict(self, x):
        return [ self._predictSingle(sample) for sample in x ]

