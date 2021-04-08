import utils
import numpy as np

class DataProcessor():
    
    def __init__(self, filePath):
        data = np.genfromtxt(filePath, delimiter=',')
        splitData = self.splitData(data)
        self.training = splitData[0]
        self.testing = splitData[1]

    def splitData(self, data):
        np.random.shuffle(data)
        seventyPercent = int(np.floor(0.75*len(data[:])))
        return [data[:seventyPercent], data[seventyPercent:]]
      

def createDataset(dataDirectory):
    pass

def calculateMacroRecall(actuals, predictions):
    pass

def calculateMacroPrecision(actuals, predictions):
    pass

def calculateMacroF1(actuals, predictions):
    pass