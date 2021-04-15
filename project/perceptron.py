from data_processing import DataProcessor
import numpy as np

class Perceptron():

    def __init__(self, learningRate, classLabelValue, weights=None, ):
        self.eta = learningRate
        self.clv = classLabelValue
        self.dp = DataProcessor(classLabelValue)
        if weights is None:
            self.generateRandomWeights(0.005, 0.01)

    def generateRandomWeights(self, min, max):
        self.w = (np.random.uniform(min, max, size=(DataProcessor.featureLength)))

    def train(self, epochs):
        print("starting training")
        accuracies = []

        for i in range(0, epochs):
            print("epoch: {}".format(i))
            predicitions = []

            # loop over the training set
            # len(self.dp.training)
            for ii in range(0, len(self.dp.training)):
                trainingExample = self.dp.training[ii]
               
                # call the predict
                [predicition, confidence] = self.predict(example=trainingExample[:-1])
                predicitions.append(predicitions)
  
                # if not call update weights
                self.updateWeights(trainingExample[-1], predicition, np.copy(trainingExample[:-1]))
                    

            # accuracy = DataProcessor.getAccuracy(np.array(predicitions), self.dp.training[:, -1])
            # accuracies.append(accuracy)

        print("starting training ended weights are: ")
        print(self.w)
        return np.array(accuracies)

    def updateWeights(self, realOutput, perceptronOutput, featuresCopy):
        featuresCopy = np.insert(featuresCopy, 0, 1, axis=0)
        self.w = self.w + self.eta*(realOutput-perceptronOutput)*featuresCopy
    
    # receives examples in 1d array 
    def predict(self, example):
        if(self.w is None):
            return "perceptron not trained"
        confidence = np.dot(self.w[1:], example)
        confidence = confidence + self.w[0]
        print("confidence; {}".format(confidence))
        if confidence>0:
            return [self.clv, confidence]
        return -1



    def test(self):
        pass

DataProcessor.initializeDataProcessorClass()
applePerceptron = Perceptron(learningRate=0.1, classLabelValue=0, )
accuracies = applePerceptron.train(1)
print(accuracies)

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