from data_processing import DataProcessor

class Perceptron():

    def __init__(self, learningRate, weights, classLabelValue):
        self.eta = learningRate
        self.w = weights
        self.clv = classLabelValue
        self.dp = DataProcessor(classLabelValue)

    # todo: should receive the data and saves the weights in a csv file
    def train(self, epochs):
        
        pass
    
    # todo: should predict 0 or 1 if the example suits the class or not along with the sureness (?)
    def predict(self, example):
        if(self.w is None):
            return "perceptron not trained"
    