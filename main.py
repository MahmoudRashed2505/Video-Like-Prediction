from MileStone1.Model1 import Model1
from MileStone1.Model2 import Model2
from MileStone2.Classification import Classification
from PreProcessing.process import process
from os import system
import pandas as pd

system('cls')
art = '''
  __  __ _       ____            _           _   
 |  \/  | |     |  _ \ _ __ ___ (_) ___  ___| |_ 
 | |\/| | |     | |_) | '__/ _ \| |/ _ \/ __| __|
 | |  | | |___  |  __/| | | (_) | |  __/ (__| |_ 
 |_|  |_|_____| |_|   |_|  \___// |\___|\___|\__|
                              |__/                                                                                                            
'''


print("\n\n"+"\033[1;32;40m"+art+"\033[0;37;40m")

print("\n\n"+"\033[1;32;40m"+"Welcome to the Machine Learning Project"+"\033[0;37;40m")

# Ask the user to enter the path of the data file or use the default path
print("\033[1;32;40m"+"Enter or Drag the path of the data file or Leave Empty to use the default Datasets"+"\033[0;37;40m")
path = input("\033[1;32;40m"+"Path: "+"\033[0;37;40m")
if path == "":
    MileStone1Path = "Data/VideoLikesDataset.csv"
    MileStone2Path = "Data/VideoLikesDatasetClassification.csv"
while True:
    system('cls')
    print("\n\n"+"\033[1;32;40m"+art+"\033[0;37;40m")
    print("\n\n"+"\033[1;32;40m"+"Please select the MileStone you want to run"+"\033[0;37;40m")
    print("\n"+"\033[1;32;40m"+"1. MileStone1"+"\033[0;37;40m")
    print("\033[1;32;40m"+"2. MileStone2"+"\033[0;37;40m")
    print("\033[1;32;40m"+"3. Change Dataset"+"\033[0;37;40m")
    print("\033[1;32;40m"+"4. Exit"+"\033[0;37;40m")
    choice = int(input("\n\n"+"\033[1;32;40m"+"Enter your choice : "+"\033[0;37;40m"))
    
    # If User Choosed Milestone 1
    if choice == 1:
        
        if path == "":
            data_file = pd.read_csv(MileStone1Path)
        else:
            data_file = pd.read_csv(path)
        
        X_train, X_test, y_train, y_test = process(data_file,MileStone1=True)
        system('cls')
        print("\n\n"+"\033[1;32;40m"+art+"\033[0;37;40m")
        Model1(X_train,y_train,X_test,y_test)
        input("\033[1;32;40m"+"Press Enter to continue..."+"\033[0;37;40m")
        system('cls')
        print("\n\n"+"\033[1;32;40m"+art+"\033[0;37;40m")
        Model2(X_train,y_train,X_test,y_test)
        input("\033[1;32;40m"+"Press Enter to continue..."+"\033[0;37;40m")
    
    # If User Choosed Milestone 2
    elif choice == 2:
        
        if path == "":
                data_file = pd.read_csv(MileStone2Path)
        else:
            data_file = pd.read_csv(path)
        
        print("\n\n"+"\033[1;32;40m"+"Do you want to use Automatic Hypertune ? (Y/[N])"+"\033[0;37;40m")
        hypertune_choice = input("\033[1;32;40m"+"Answer: "+"\033[0;37;40m")
        
        if hypertune_choice.lower() == 'y':
            hypertune = True
        else:
            hypertune = False
                
        X_train, X_test, y_train, y_test = process(data_file,MileStone1=False)
        system('cls')
        print("\n\n"+"\033[1;32;40m"+art+"\033[0;37;40m")
        Classification(X_train,y_train,X_test,y_test,hypertune=hypertune)
        input("\033[1;32;40m"+"Press Enter to continue..."+"\033[0;37;40m")
    
    # If User Choosed to Change Dataset
    elif choice == 3:
        print("\n\n"+"\033[1;32;40m"+art+"\033[0;37;40m")
        data_file = pd.read_csv(input("\n\n"+"\033[1;32;40m"+"Enter the path of the CSV File Or Drag it Here : "+"\033[0;37;40m"))
    
    # If User Choosed to Exit
    elif choice == 4:
        # Exit the application
        exit()
    
    # If User Choosed an Invalid Choice
    else:
        print("\n\n"+"\033[1;32;40m"+"Please select a valid option"+"\033[0;37;40m")
        input("\033[1;32;40m"+"Press Enter to continue..."+"\033[0;37;40m")