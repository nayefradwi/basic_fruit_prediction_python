from perceptron import Perceptron

class Main():
    pass
    
    # todo: for each weight create a perceptron with its corresponding classLabel
    def __init__(self):
        pass
    
    # trains a collection of perceptrons 
    def train(self, path):
        # create data set
        # 
        print(path)
    
    
    def predictOne(self, path):
        print(path)

    def predictMultiple(self, path):
        print(path)

main = Main()
while(True):
    choice = input("choose:\n1) train on a new dataset\n2) predict\n3) quit\n")
    
    if(choice == "3"):
        exit(1)
    elif(choice == "2"):
        main.predictOne("predicted apple")
    elif(choice == "1"):
        path = input("put the path of the data set folder:\n")
        main.train("path")
    else:
        print("unknown command")
        