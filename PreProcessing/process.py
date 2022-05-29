from PreProcessing.DataCleaning import clean_data,Feature_Encoder
from PreProcessing.Stats import get_stats_for_entire_dataframe,get_top_features
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale # Used for Scaling the data
from os import system
import numpy

art = '''
  __  __ _       ____            _           _   
 |  \/  | |     |  _ \ _ __ ___ (_) ___  ___| |_ 
 | |\/| | |     | |_) | '__/ _ \| |/ _ \/ __| __|
 | |  | | |___  |  __/| | | (_) | |  __/ (__| |_ 
 |_|  |_|_____| |_|   |_|  \___// |\___|\___|\__|
                              |__/                                                                                                            
'''

def process(df,MileStone1 = True):
    
    # Get descriptive stats on all the columns
    system('cls')
    print("\n\n"+"\033[1;32;40m"+art+"\033[0;37;40m")
    get_stats_for_entire_dataframe(df)
    input("\033[1;32;40m"+"Press Enter to continue..."+"\033[0;37;40m")
    
    
    # Clean the data
    system('cls')
    print("\n\n"+"\033[1;32;40m"+art+"\033[0;37;40m")
    print("\n\n"+"\033[1;32;40m"+"Cleaning the Data......"+"\033[0;37;40m")
    cleaned_data = clean_data(df)
    
    

    if MileStone1:
        print("\033[1;32;40m"+"Get Correlation Between data and extract top features......"+"\033[0;37;40m")
        # Get Correlation between Columns
        correlation = cleaned_data.corr()
        features = cleaned_data[get_top_features(correlation,cleaned_data)]
    
    else:
        
        print("\033[1;32;40m"+"Feature Encoding for (Video Popularity)......"+"\033[0;37;40m")

        for index,row in cleaned_data.iterrows():
            if cleaned_data.loc[index,'VideoPopularity']=='Medium':
                cleaned_data.loc[index,'VideoPopularity'] = 0
            elif cleaned_data.loc[index,'VideoPopularity'] == 'High':
                cleaned_data.loc[index,'VideoPopularity'] = 1
            elif cleaned_data.loc[index,'VideoPopularity'] == 'Low':
                cleaned_data.loc[index,'VideoPopularity'] = -1
    
    # Split Features and Labels
    features = cleaned_data.iloc[:,:-1]
    target = cleaned_data.iloc[:,-1]
    
    if MileStone1 == False:
        target = target.astype(int)
    
    #Feature Scaling
    print("\033[1;32;40m"+"Feature Scaling......"+"\033[0;37;40m")
    features[['views','comment_count']] = minmax_scale(features[['views','comment_count']])


    print("\033[1;32;40m"+"Splitting the data into training and testing data......"+"\033[0;37;40m")
    input("\033[1;32;40m"+"Press Enter to continue..."+"\033[0;37;40m")
    # Split the data into 80% training and 20% testing data
    return train_test_split(features,target,test_size=0.2,shuffle=True,random_state=10)
    
    
    