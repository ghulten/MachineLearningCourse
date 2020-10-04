# some environments have trouble with text encodings, try:
#   f = open(path, 'r', encoding='unicode_escape')

import collections
import os

kDataPath = os.path.join("MachineLearningCourse", "MLProjectSupport", "Adult", "dataset", "adult.data")

def LoadRawData(path=kDataPath):
    print("Loading data from: %s" % os.path.abspath(path))
    f = open(path, 'r')
    
    lines = f.readlines()

    kNumberExamplesExpected = 31561

    if(len(lines) != kNumberExamplesExpected):
        message = "Attempting to load %s:\n" % (path)
        message += "   Expected %d lines, got %d.\n" % (kNumberExamplesExpected, len(lines))
        message += "    Check the path to training data and try again."
        raise UserWarning(message)

    x = []
    y = []

    for l in lines:
        rawFields = l.split(",")
        if(len(rawFields) != 15):
            message = "Expected 15 fields, found %d in: %s" % (len(fields), str(l))
            raise UserWarning(message)

        fields = [ x.strip() for x in rawFields ]

        if(fields[-1] == "<=50K"):
            y.append(0)
            x.append(fields[0:-1])
        elif(fields[-1] == ">50K"):
            y.append(1)
            x.append(fields[0:-1])
        else:
            message = "Attempting to process %s\n" % (l)
            message += "   Did not match expected format."
            message += "    Check the path to training data and try again."
            raise UserWarning(message)

    return (x, y)