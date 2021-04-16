from data_processing import DataProcessor
import numpy as np
import matplotlib.pyplot as plt

class Perceptron():

    # list of perceptron objects
    perceptrons = []

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

    def getLabelNameFromLabelValue(self):
        return DataProcessor.classLabelsList[self.clv]

    def updateWeights(self, realLabel, perceptronOutput, featuresCopy):
        featuresCopy = np.insert(featuresCopy, 0, 1, axis=0)
        self.w = self.w + self.eta*(realLabel-perceptronOutput)*featuresCopy

    def train(self, epochs, addGraph=False, runValidation=False):
        print("started training!")
        validationAccuracies = []
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
            if runValidation:
                validationAccuracies.append(self.validation())

        if addGraph:
            plt.figure(1)
            plt.plot(accuracies, label=self.getLabelNameFromLabelValue())
            plt.title("accuracy vs number of epochs")
            if runValidation:
                plt.figure(2)
                plt.plot(validationAccuracies, label=self.getLabelNameFromLabelValue())
                plt.title("validation accuracy vs number of epochs")
            plt.legend()
        return np.array(accuracies)

    def validation(self):
        predicitions = []
        
        for validationExample in self.dp.validation:
            realLabel = self.getZeroOrOneLabel(validationExample[-1])
            [predicition, confidence] = self.predict(example=validationExample[:-1])
            predicitions.append(predicition)
            self.updateWeights(realLabel, predicition, np.copy(validationExample[:-1]))

        predicitions = np.array(predicitions)
        trainingLabels = np.copy(self.dp.validation[:, -1])
        trainingLabels = np.vectorize(self.getZeroOrOneLabel)(trainingLabels)
        return DataProcessor.getAccuracy(np.array(predicitions),trainingLabels)

    def trainModel(runValidation = False):

        # loading trainingset and testing set
        DataProcessor.initializeDataProcessorClass()
        # define a standard learning rate that each perceptron follows
        learningRate = 0.01
        # define a 2d np array of weights that will then be stored in a csv file
        weights = []
        # create a standard number of epochs
        epochs = 100

        for i in range(0, DataProcessor.numberOfClasses):
            # create a perceptron with clv = i
            p = Perceptron(learningRate,classLabelValue=i)
            print("training {} perceptron".format(p.getLabelNameFromLabelValue()))
            # train perceptron
            p.train(epochs, addGraph=True, runValidation=runValidation)
            # append weights
            weights.append(p.w)
            # append perceptron
            Perceptron.perceptrons.append(p)

        # store weights in a csv file 
        weights = np.array(weights)
        np.savetxt("weights.csv", weights, delimiter=",")
        # display a plot that shows accuracy vs epoch for multiple perceptrons
        plt.show()

  
        
    def predict(self, example):
        if(self.w is None):
            return "perceptron not trained"
        confidence = np.dot(self.w[1:], example)
        confidence = confidence + self.w[0]
        if confidence>0:
            return [1, confidence]
        return [0, confidence]

    def predictModel():
        # create perceptrons if perceptron list is empty
        # use each one to predict and append the prediction
        # get where prediction == 1 (indexes),
        # sort the answer based on highest confidence
        # then you can return both label name and index (predicition)
        pass

Perceptron.trainModel(runValidation=True)


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