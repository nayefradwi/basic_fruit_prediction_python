from utils import get_image_3d
from data_processing import DataProcessor
import numpy as np
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename

'''
perceptron class, this is the main class for creating the model

static attributes:
perceptrons - list of trained perceptrons
'''
class Perceptron():

    # list of perceptron objects
    perceptrons = []
    isDpInitialized = False
    '''
    Initialize will initialize the model by creating preceptrons 
    equal to the number of classes defined in the classLabels.csv,

    each perceptron will be assigned by its corresponding weights from weights.csv

    data processor needs to be initialized as well so that the class labels are
    loaded

    return:
    0 - means model initialized successfully (deployed / ready to use in application)
    1 - means that the model has not been trained and hence there are no weights.csv
    '''
    def initialize():
        DataProcessor.initializeDataProcessorClass()
        try:
            classesWeights = np.genfromtxt("weights.csv",  delimiter=',')
            for i in range(0, len(classesWeights)):
                Perceptron.perceptrons.append(Perceptron(None, classLabelValue=i, weights=classesWeights[i]))
        except Exception as e:
            print(e.with_traceback())
            print("no weights csv")
            return -1
        return 0


    '''
    the constructor creates a perceptron object to be used later

    attributes:
    eta - the learning rate of the perceptrons algorihm
    clv - the class label value (0, 1 ,2 etc)
    dp - instance of DataProcessor so each perceptron can have its
    training and validation set
    w - weights of the perceptron which will be used for classification
    '''
    def __init__(self, learningRate, classLabelValue, weights=None, ):
        self.eta = learningRate
        self.clv = classLabelValue
        self.dp = DataProcessor(classLabelValue)
        if weights is None:
            self.generateRandomWeights(0.005, 0.01)
        else:
            self.w = weights

    '''
    this is called incase a perceptron instance is created without weights
    this will be used for when the perceptrons have no initial weights and it
    is about to be trained

    params:
    min - lower bound for the random weights
    max - upper bound for the random weights

    return:
    None, it reassigns the weights attribute
    '''
    def generateRandomWeights(self, min, max):
        self.w = (np.random.uniform(min, max, size=(DataProcessor.featureLength)))
    
    '''
    this is used when training a perceptron to make its class = 1
    and all other classes = 0

    params:
    label - 0 to N based on how many classes are there, its the class' label
    '''
    def getZeroOrOneLabel(self,label):
        return int(label == self.clv)

    '''
    if data processor is initialized this will get the class label based on
    the object's clv
    '''
    def getLabelNameFromLabelValue(self):
        return DataProcessor.classLabelsList[self.clv]

    '''
    used in training to update the weights of a perceptron instance

    params:
    realLabel - the examples actual label (0 or 1)
    perceptronOutput - the classification done by the perceptron (0 or 1)
    featuresCopy - a copy of the array of features of the example that was used to get
    the perceptron output
    '''
    def updateWeights(self, realLabel, perceptronOutput, featuresCopy):
        featuresCopy = np.insert(featuresCopy, 0, 1, axis=0)
        self.w = self.w + self.eta*(realLabel-perceptronOutput)*featuresCopy

    '''
    this function trains one perceptron by looping over the training array in the 
    data processor instance of this perceptron. it loops over the training array
    epochs times, then reassigns the w attribute of the perceptron instance

    params:
    epochs - how many times to loop over the training (int)
    addGraph - display accuracies in a graph (boolean)
    runValidation - run validation after each epoch and display a graph (boolean)

    return:
    accuracies - a list of accuracies with size "epochs"
    '''
    def train(self, epochs, addGraph=False, runValidation=False):
        print("started training!")
        validationAccuracies = []
        accuracies = []
        for i in range(0, epochs):
            print("epoch: {}".format(i))
            predicitions = []
            
            # loop over the training set
            for trainingExample in DataProcessor.globalTrainingSet:
                realLabel = self.getZeroOrOneLabel(trainingExample[-1])
                [predicition, confidence] = self.predict(example=trainingExample[:-1])
                predicitions.append(predicition)
                self.updateWeights(realLabel, predicition, np.copy(trainingExample[:-1]))

            predicitions = np.array(predicitions)
            trainingLabels = np.copy(DataProcessor.globalTrainingSet[:,-1])
            trainingLabels = np.vectorize(self.getZeroOrOneLabel)(trainingLabels)
            accuracy = DataProcessor.getAccuracy(np.array(predicitions),trainingLabels)
            accuracies.append(accuracy)
            if runValidation:
                validationAccuracies.append(self.validation())
            np.random.shuffle(DataProcessor.globalTrainingSet)
        if addGraph:
            plt.figure(1)
            plt.plot(accuracies, label=self.getLabelNameFromLabelValue())
            plt.title("accuracy vs number of epochs")
            plt.legend()
            if runValidation:
                plt.figure(2)
                plt.plot(validationAccuracies, label=self.getLabelNameFromLabelValue())
                plt.title("validation accuracy vs number of epochs")
                plt.legend()
        return np.array(accuracies)

    '''
    loops over the validation array that is in the data processor instance
    of this perceptron instance. loop is done once 

    return:
    accuracy - accuracy of the validation run
    '''
    def validation(self):
        predicitions = []
        
        for validationExample in self.dp.validation:
            realLabel = self.getZeroOrOneLabel(validationExample[-1])
            [predicition, confidence] = self.predict(example=validationExample[:-1])
            predicitions.append(predicition)

        predicitions = np.array(predicitions)
        trainingLabels = np.copy(self.dp.validation[:, -1])
        trainingLabels = np.vectorize(self.getZeroOrOneLabel)(trainingLabels)
        return DataProcessor.getAccuracy(np.array(predicitions),trainingLabels)


    '''
    trains the entire model by creating N perceptrons based on the numberOfClasses available
    each perceptron will be trained by calling the train instance method

    it will display a graph after training is done

    params:
    epochs - number of epochs all perceptrons being trained
    learningRate - learning rate of all the perceptrons being trained    
    runValidation - to be passed to the train function

    return:
    None, but a file is created called weights.csv which has the weights of all the perceptrons
    this makes the model deployable 
    '''
    def trainModel(epochs, learningRate,runValidation = False):
        if not Perceptron.isDpInitialized:
            DataProcessor.initializeDataProcessorClass()
        # define a 2d np array of weights that will then be stored in a csv file
        weights = []

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

  
    '''
    uses the perceptron's weights attribute to compute the dot product
    
    returns:
    [classification, confidence] - classification is 1 or 0, confidence is the dot product value 
    '''
    def predict(self, example):
        if(self.w is None):
            return "perceptron not trained"
        confidence = np.dot(self.w[1:], example)
        confidence = confidence + self.w[0]
        if confidence>0:
            return [1, confidence]
        return [0, confidence]
 
    '''
    this will call the predict instance method of all initialized perceptrons,
    it will get the index of the predicition that has 1 and the highest confidence

    params:
    example - 1d array of the features of the example that is going to be classified
    '''
    def predictModel(example):
        # create perceptrons if perceptron list is empty
        if len(Perceptron.perceptrons) == 0:
            return "no model trained or initialized"
            
        perceptronPredictions = []
        for p in Perceptron.perceptrons:
            # use each one to predict and append the prediction
            perceptronPredictions.append(p.predict(example))
        perceptronPredictions = np.array(perceptronPredictions)
        # get where prediction == 1 (indexes)
        # sort the answer based on highest confidence
        onesIndex = np.argmax(perceptronPredictions[:,-1], axis=0)
        label = "unable to find label due to not loading classlabel.csv"
        try:
            label = Perceptron.perceptrons[onesIndex].getLabelNameFromLabelValue()
        except:
            pass
        # then you can return both label name and index (predicition)
        return [onesIndex, label]
    
    def testModel():
        precisions = []
        recalls = []
        for i in range(0, len(Perceptron.perceptrons)):
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            print("testing class: {}".format(i))
            # testingSetFiltered = DataProcessor.globalTestingSet[np.where(DataProcessor.globalTestingSet[:, -1]==i)]
            for testExample in DataProcessor.globalTestingSet:
                [label, labelName] =Perceptron.predictModel(testExample[:-1])
                if label == i and testExample[-1] == i:
                    tp = tp + 1
                    continue
                elif label != i and testExample[-1] == i:
                    fn = fn + 1
                    continue
                elif label != i and testExample[-1] != i:
                    tn = tn +1
                    continue
                elif label == i and testExample[-1] != i:
                    fp = fp + 1
                    continue
            # get recall
            recalls.append(DataProcessor.getRecall(tp, fn))
            # get percision
            precisions.append(DataProcessor.getPrecision(tp, fp))
        precision = np.average(precisions)
        recall = np.average(recalls)
        print("percision of class: {}; recall: {}".format(precision,recall))


        
# Perceptron.trainModel(epochs=1000,learningRate=0.01, runValidation=False)
# Perceptron.testModel()
status = Perceptron.initialize()
if status != 0:
    print("not trained before")
    exit(0)
while True:
    filename = askopenfilename(filetypes=[("images", "*.jpg")])
    image3d = get_image_3d(filename)
    imageFeature = DataProcessor.featureExtraction(image3d)
    predicition = Perceptron.predictModel(example=imageFeature)
    print(predicition)


