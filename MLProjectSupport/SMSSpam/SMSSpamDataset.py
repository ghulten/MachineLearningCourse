# some environments have trouble with text encodings, try:
#   f = open(path, 'r', encoding='unicode_escape')

import os

kDataPath = os.path.join("MachineLearningCourse", "MLProjectSupport", "SMSSpam", "dataset", "SMSSpamCollection")

def LoadRawData(path=kDataPath):
    print("Loading data from: %s" % os.path.abspath(path))
    f = open(path, 'r')
    
    lines = f.readlines()

    kNumberExamplesExpected = 5474

    if(len(lines) != kNumberExamplesExpected):
        message = "Attempting to load %s:\n" % (path)
        message += "   Expected %d lines, got %d.\n" % (kNumberExamplesExpected, len(lines))
        message += "    Check the path to training data and try again."
        raise UserWarning(message)

    x = []
    y = []

    for l in lines:
        if(l.startswith('ham')):
            y.append(0)
            x.append(l[4:].strip())
        elif(l.startswith('spam')):
            y.append(1)
            x.append(l[5:].strip())
        else:
            message = "Attempting to process %s\n" % (l)
            message += "   Did not match expected format."
            message += "    Check the path to training data and try again."
            raise UserWarning(message)

    return (x, y)




