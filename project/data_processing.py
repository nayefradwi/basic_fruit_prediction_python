import matplotlib.pyplot as plt
import utils
import os
import numpy as np

'''
data processor class handles things such as creating training and testset,
creating training and validation sets for a specific perceptron, and other
data processing functions

static attributes:
numberOfClasses - the number of classes that are used in the model
classLabelsList - list that is used to map an integer to a string (clv to class label)
globalTrainingSet - the entire training set array 
globalTestingSet - the entire testing set array
featureLength - the length of features per sample
'''
class DataProcessor():
    numberOfClasses = 0
    classLabelsList =[]
    globalTrainingSet = []
    globalTestingSet = []
    featureLength = 0

    '''
    data processor instance is used by perceptron instances to have access to a specific 
    undersampled data set for the class that the perceptron is assigned to 

    attributes:
    training - the training set
    validation - the validation set 
    '''
    def __init__(self, classLabelValue):
        try:
            data = DataProcessor.getTrainingSetForClass(classLabelValue)
            self.training = data[0]
            self.validation = data[1]
        except:
            pass


    '''
    gets the training set and validation set for a specific class

    params:
    classLabelValue - 0, 1, 2, etc.
    
    return:
    validation set and training set
    '''   
    def getTrainingSetForClass(classLabelValue):
        # get only classlabel==classLabelValue from data
        classData = DataProcessor.globalTrainingSet[np.where(DataProcessor.globalTrainingSet[:, -1] == classLabelValue)]

        # get classlabel!=classLabelValue from data
        nonClassData = DataProcessor.globalTrainingSet[np.where(DataProcessor.globalTrainingSet[:, -1] != classLabelValue)]
        
        # undersampling
        nonClassData = nonClassData[:len(classData)]
        
        # returning training and validation set
        dataSetForOneClass = np.append(classData, nonClassData, axis=0)
        return DataProcessor.splitData(np.copy(dataSetForOneClass),fraction=1)
        

    '''
    loads up class labels, training set, testing set, (if dataset is already created)
    this is extremely necessary for either:
    a) you want to train perceptrons
    b) you want to test perceptrons

    the training set and testing set are not required if the model is deployed
    '''
    def initializeDataProcessorClass():
        try:
            DataProcessor.classLabelsList = open("classLabels.csv", "r").read().split('\n')[:-1]
            DataProcessor.numberOfClasses = len(DataProcessor.classLabelsList)
        except:
            print("there are no classes that can be used in the model")
        try:
            DataProcessor.globalTrainingSet = np.genfromtxt('training.csv', delimiter=',')
            DataProcessor.globalTestingSet = np.genfromtxt('testing.csv', delimiter=',')
            DataProcessor.featureLength = DataProcessor.globalTrainingSet.shape[-1]
        except:
            print("dataset has not been created please, create a dataset before running training")

    '''
    splits the data up into 2 different arrays
    
    params:
    data - numpy array to split
    fraction - fraction on how much to split the data (float)

    return:
    the 2 parts after the split
    '''
    def splitData(data, fraction=0.75):
        np.random.shuffle(data)
        seventyPercent = int(np.floor(fraction*len(data[:])))
        return [np.array(data[:seventyPercent]), np.array(data[seventyPercent:])]

    '''
    loops over the images, channels and calls the necessary functions to extract the 
    features from the images.

    currently it takes the average of x by x areas of the image

    params:
    image3d - (x,y,3) rgb image array

    return:
    flattened - the features flattened to be an 1d array
    '''
    def featureExtraction(image3d):
        flattened = np.array([])

        # loop every channel
        for rgbChannelIndex in range(0,image3d.shape[-1]):
            averageMatrixOFChannel = DataProcessor.getAverageChannelMatrix(image3d[:,:, rgbChannelIndex], stepSize=10)
            flattened = np.append(flattened, averageMatrixOFChannel)
            varianceOfChannel = DataProcessor.getVarianceOfChannel(image3d[:,:, rgbChannelIndex], stepSize=10)
            flattened = np.append(flattened, varianceOfChannel)
            varianceOfChannel = DataProcessor.getStdOfChannel(image3d[:,:, rgbChannelIndex], stepSize=10)
            flattened = np.append(flattened, varianceOfChannel)
            maxOfChannel = DataProcessor.getMaxOfChannel(image3d[:,:, rgbChannelIndex], stepSize=10)
            flattened = np.append(flattened, maxOfChannel)
            minOfChannel = DataProcessor.getMinOfChannel(image3d[:,:, rgbChannelIndex], stepSize=10)
            flattened = np.append(flattened, minOfChannel)
        
        flattened = DataProcessor.absolute_scale(flattened);
        return flattened

    '''
    filters the matrix based on step size and returns the average
    ''' 
    def getAverageChannelMatrix(matrix, stepSize):
        averageChannelMatrix = np.empty((int(matrix.shape[0]/stepSize), int(matrix.shape[1]/stepSize)))
        for i in range(0, averageChannelMatrix.shape[0]):
            for ii in range(0, averageChannelMatrix.shape[1]):
                average = DataProcessor.getAverage(matrix[i:(i+1)*stepSize, ii:(ii+1)*stepSize])
                averageChannelMatrix[i, ii] = average.astype(int)
        return averageChannelMatrix.reshape(averageChannelMatrix.shape[0]*averageChannelMatrix.shape[1])

    '''
    filters the matrix based on step size and returns the variance
    ''' 
    def getVarianceOfChannel(matrix, stepSize):
        varianceChannelMatrix = np.empty((int(matrix.shape[0]/stepSize), int(matrix.shape[1]/stepSize)))
        for i in range(0, varianceChannelMatrix.shape[0]):
            for ii in range(0, varianceChannelMatrix.shape[1]):
                average = DataProcessor.getVariance(matrix[i:(i+1)*stepSize, ii:(ii+1)*stepSize])
                varianceChannelMatrix[i, ii] = average.astype(int)
        return varianceChannelMatrix.reshape(varianceChannelMatrix.shape[0]*varianceChannelMatrix.shape[1])  
    '''
    filters the matrix based on step size and returns the std
    ''' 
    def getStdOfChannel(matrix, stepSize):
        varianceChannelMatrix = np.empty((int(matrix.shape[0]/stepSize), int(matrix.shape[1]/stepSize)))
        for i in range(0, varianceChannelMatrix.shape[0]):
            for ii in range(0, varianceChannelMatrix.shape[1]):
                average = DataProcessor.getVariance(matrix[i:(i+1)*stepSize, ii:(ii+1)*stepSize])
                varianceChannelMatrix[i, ii] = average.astype(int)
        return varianceChannelMatrix.reshape(varianceChannelMatrix.shape[0]*varianceChannelMatrix.shape[1])  
    '''
    filters the matrix based on step size and returns min
    ''' 
    def getMinOfChannel(matrix, stepSize):
        minChannelMatrix = np.empty((int(matrix.shape[0]/stepSize), int(matrix.shape[1]/stepSize)))
        for i in range(0, minChannelMatrix.shape[0]):
            for ii in range(0, minChannelMatrix.shape[1]):
                average = DataProcessor.getMin(matrix[i:(i+1)*stepSize, ii:(ii+1)*stepSize])
                minChannelMatrix[i, ii] = average.astype(int)
        return minChannelMatrix.reshape(minChannelMatrix.shape[0]*minChannelMatrix.shape[1]) 
    
    '''
    filters the matrix based on step size and returns max 
    ''' 
    def getMaxOfChannel(matrix, stepSize):
        maxChannelMatrix = np.empty((int(matrix.shape[0]/stepSize), int(matrix.shape[1]/stepSize)))
        for i in range(0, maxChannelMatrix.shape[0]):
            for ii in range(0, maxChannelMatrix.shape[1]):
                average = DataProcessor.getMax(matrix[i:(i+1)*stepSize, ii:(ii+1)*stepSize])
                maxChannelMatrix[i, ii] = average.astype(int)
        return maxChannelMatrix.reshape(maxChannelMatrix.shape[0]*maxChannelMatrix.shape[1])

    '''
    perform mathematical operations on
    a matrix

    params:
    matrix - numpy array

    '''
    def getAverage(matrix):
        return np.mean(matrix)
    
    def getVariance(matrix):
        return np.var(matrix)

    def getVariance(matrix):
        return np.std(matrix)
    
    def getMax(matrix):
        return np.max(matrix)

    def getMin(matrix):
        return np.min(matrix)

    '''
    gets the accuracy 

    params:
    predictions - 1d array of predictions
    realOutput - 1d array of the real labels as 1 or 0

    return: 
    accuracy - right/total
    '''
    def getAccuracy(predictions, realOutput):
        return len(predictions[np.where(predictions==realOutput)])/len(realOutput)


    def getPrecision(tp, fp):
        try:
            return tp/(tp+fp)
        except:
            return 0
    
    def getRecall(tp, fn):
        try:
            return tp/(tp+fn)
        except:
            return 0
    '''
    params:
    features - 1d array of features without the label

    return:
    scaled features (1d array)
    '''
    def absolute_scale(features):
        return features/np.max(features)

    '''
    loops over a directory that has subdirectories
    
    each subDirectory will be a class
    
    will loop over each file in a sub directory, extract its features then adds the label to the end of the features array. 
    the features 1d array is then appended to imagesFeatures so that all the images can then be converted to a csv file.

    after the loop the imagesFeatures array is split and 2 csv files are created
    1) training.csv
    2) testing.csv

    this should be called only once, and that is to create the dataset for training.
    this is not necessary if the model has already been trained and a weights.csv & classLabels.csv files already
    exist

    params:
    dataDirectoryPath - the directory that has the class sub-directories

    '''
    def createDataset(dataDirectoryPath):
        directories = os.listdir(dataDirectoryPath)
        
        classes = []
        imagesFeatures = []
        for i in range(0, len(directories)):
            classDirectory = directories[i]
            
            print("directory: {}".format(classDirectory))
            classes.append(classDirectory)
            
            classDirectoryPath = "{}/{}".format(dataDirectoryPath,classDirectory)

            if os.path.isdir(classDirectoryPath):
                imageFileNames = os.listdir(classDirectoryPath)

                for ii in range(0, len(imageFileNames)):
                    imageFile = imageFileNames[ii]
                    imageFilePath = "{}/{}/{}".format(dataDirectoryPath,classDirectory,imageFile)

                    if os.path.isfile(imageFilePath):
                        image3dArray = utils.get_image_3d(imageFilePath)
                        features = DataProcessor.featureExtraction(image3dArray)
                        features = np.append(features,[i])
                        imagesFeatures.append(features)

        imagesFeatures = np.array(imagesFeatures)
        [training, testing] = DataProcessor.splitData(imagesFeatures)
        np.savetxt("training.csv", training, delimiter=",")
        np.savetxt("testing.csv", testing, delimiter=",")
        np.savetxt("classLabels.csv", np.array(classes), delimiter=',',fmt="%s")
        return imagesFeatures

# DataProcessor.createDataset("./data")