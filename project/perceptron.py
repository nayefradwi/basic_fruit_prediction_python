from data_processing import DataProcessor
import numpy as np
import matplotlib.pyplot as plt

class Perceptron():

    def __init__(self, learningRate, classLabelValue, weights=None, ):
        self.eta = learningRate
        self.clv = classLabelValue
        self.dp = DataProcessor(classLabelValue)
        if weights is None:
            self.generateRandomWeights(0.005, 0.01)
        else:
            self.w = weights

    def generateRandomWeights(self, min, max):
        self.w = (np.random.uniform(min, max, size=(DataProcessor.featureLength)))
    
    def getZeroOrOneLabel(self,label):
        return int(label == self.clv)

    def train(self, epochs):
        print("started training!")
        accuracies = []
        for i in range(0, epochs):
            print("epoch: {}".format(i))
            predicitions = []
            
            # loop over the training set
            for trainingExample in self.dp.training:

                realLabel = self.getZeroOrOneLabel(trainingExample[-1])
                [predicition, confidence] = self.predict(example=trainingExample[:-1])
                predicitions.append(predicition)
                self.updateWeights(realLabel, predicition, np.copy(trainingExample[:-1]))

            predicitions = np.array(predicitions)
            trainingLabels = np.copy(self.dp.training[:, -1])
            trainingLabels = np.vectorize(self.getZeroOrOneLabel)(trainingLabels)
            accuracy = DataProcessor.getAccuracy(np.array(predicitions),trainingLabels)
            accuracies.append(accuracy)
            
        return np.array(accuracies)

    def updateWeights(self, realLabel, perceptronOutput, featuresCopy):
        featuresCopy = np.insert(featuresCopy, 0, 1, axis=0)
        self.w = self.w + self.eta*(realLabel-perceptronOutput)*featuresCopy
    
    # receives examples in 1d array 
    def predict(self, example):
        if(self.w is None):
            return "perceptron not trained"
        confidence = np.dot(self.w[1:], example)
        confidence = confidence + self.w[0]
        if confidence>0:
            return [1, confidence]
        return [0, confidence]



    def test(self):
        pass

DataProcessor.initializeDataProcessorClass()
applePerceptron = Perceptron(learningRate=0.01, classLabelValue=0, )
accuracies = applePerceptron.train(100)
plt.plot(accuracies)
plt.title("accuracy vs number of epochs")
plt.show()
# testPerceptron = Perceptron(learningRate=0.25, classLabelValue=0, weights=np.array([0.1, 0.5, -0.4]))
# testPerceptron.updateWeights(0,1,np.array([0.5,-0.2]))
  # if predicition == 1 and trainingExample[-1] == self.clv:
                #     tp = tp + 1
                #     continue
                # elif predicition == 0 and trainingExample[-1] != self.clv:
                #     tn = tn +1
                #     continue
                # elif predicition == 0 and trainingExample[-1] == self.clv:
                #     fn = fn +1
                # elif predicition == 1 and trainingExample[-1] != self.clv:
                #     fp = fp + 1

              # tp = 0
        # tn = 0
        # fp = 0
        # fn = 0