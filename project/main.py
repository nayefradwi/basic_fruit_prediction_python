class Main():
    pass
    
    def __init__(self):
        pass

    def train(self, path):
        print(path)
    
    def predict(self, path):
        print(path)


main = Main()
while(True):
    choice = input("choose:\n1) train on a new dataset\n2) predict\n3) quit\n")
    if(choice == "3"):
        exit(1)
    elif(choice == "2"):
        main.predict("predicted apple")
    elif(choice == "1"):
        main.predict("trained on new dataset")
    else:
        print("unknown command")
        