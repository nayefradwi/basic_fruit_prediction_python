from perceptron import Perceptron
from data_processing import DataProcessor
import utils

isInitialized = False

def predict_fruit(file_path):
    global isInitialized
    if not isInitialized:
        status = Perceptron.initialize()
        if status == 0:
            print("perceptron initialized")
            isInitialized = True
        else:
            print("could not locate weights file")
            return
    image3d = utils.get_image_3d(file_path)
    imageFeature = DataProcessor.featureExtraction(image3d)
    predicition = Perceptron.predictModel(example=imageFeature)
    return predicition[0]    

# print(predict_fruit("./data/Banana/banana_0.jpg"))
# print(predict_fruit("./data/Banana/banana_0.jpg"))
