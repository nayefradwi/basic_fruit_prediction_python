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
    choice = input("choose:\n0) run main app\n1) create new dataset\n2) train again on the latest created dataset\n3) test and analyze using testset created in (1)\n4) quit\n")
    if choice == 0:
        print("main app")
    elif choice == "4":
        exit(1)
    elif choice == "3":
        main.predictOne("predicted apple")
    elif choice == "2":
        path = input("put the path of the data set folder:\n")
        main.train("path")
    elif choice == "1":
        print("created new dataset")
    else:
        print("unknown command")
        