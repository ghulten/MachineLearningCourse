# some environments have trouble with text encodings, try:
#   f = open(path, 'r', encoding='unicode_escape')

import collections
import os

def LoadKaggleData(path):
    print("Loading data from: %s" % os.path.abspath(path))
    f = open(path, 'r')
    
    lines = f.readlines()

    kNumberExamplesExpected = 100

    if(len(lines) != kNumberExamplesExpected):
        message = "Attempting to load %s:\n" % (path)
        message += "   Expected %d lines, got %d.\n" % (kNumberExamplesExpected, len(lines))
        message += "    Check the path to training data and try again."
        raise UserWarning(message)

    x = []
    IDs = []

    for l in lines:
        firstSpaceIndex = str.find(l, ' ')
        x.append(l[firstSpaceIndex + 1:])

        IDs.append(int(l[:firstSpaceIndex]))

    return (x, IDs)

def OutputSubmission(path, IDs, predictions):
    kNumberExamplesExpected = 100

    if len(IDs) != kNumberExamplesExpected or len(predictions) != kNumberExamplesExpected:
        raise UserWarning("Incorrect number of IDs or predictions. Expected %d." % (kNumberExamplesExpected))

    f = open(path, 'w')
    f.write("ID,<0/1>\n")

    for i in range(len(IDs)):
       f.write("%d, %d\n" % (IDs[i], predictions[i]))

    f.flush()
    f.close()