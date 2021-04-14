from data_processing import DataProcessor
import numpy as np

class Perceptron():

    def __init__(self, learningRate, classLabelValue, weights=None, ):
        self.eta = learningRate
        self.clv = classLabelValue
        self.dp = DataProcessor(classLabelValue)
        if weights is None:
            self.generateRandomWeights(0.02, 0.1)

    def generateRandomWeights(self, min, max):
        self.w = (np.random.uniform(min, max, size=(DataProcessor.featureLength)))

    def train(self, epochs):
        for i in range(0, epochs):
            pass

    def updateWeights(self, realOutput, perceptronOutput, featuresCopy):
        featuresCopy = np.insert(featuresCopy, 0, 1, axis=0)
        self.w = self.w + self.eta*(realOutput-perceptronOutput)*featuresCopy
    
    # receives examples in 1d array 
    def predict(self, example):
        if(self.w is None):
            return "perceptron not trained"
        

    def test(self):
        pass

DataProcessor.initializeDataProcessorClass()
applePerceptron = Perceptron(learningRate=0.1, classLabelValue=0, )
# applePerceptron.train(1)

# testPerceptron = Perceptron(learningRate=0.25, classLabelValue=0, weights=np.array([0.1, 0.5, -0.4]))
# testPerceptron.updateWeights(0,1,np.array([0.5,-0.2]))