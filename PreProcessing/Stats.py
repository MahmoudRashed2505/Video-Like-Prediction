import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def get_stats_for_entire_dataframe(df):
    
    # Print Data Frame Desctription
    print("Data Frame Describtive Stats:\n\n")
    print(df.describe(include='all'))
    
    # Print Data Frame Correlation
    print("\n\nData Frame Correlation:\n\n")
    print(df.corr())
    
    # Print Data Frame information
    print("\n\nData Frame Information:\n\n")
    print(df.info())
    
    # Print Data Frame Shape
    print("\n\nData Frame Shape:\n\n")
    print(df.shape)
    
    # Print How Many Null Values in Each Column
    print("\n\nNumber of NA Values:\n\n")
    print( df.isna().sum(axis=0))


def get_top_features(correlation,cleaned_data):
    
    #Top 50% Correlation training features with the Value
    top_features = correlation.index[abs(correlation['likes']>0.5)]
    
    #Correlation plot
    plt.subplots(figsize=(12, 8))
    top_corr = cleaned_data[top_features].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    top_features = top_features.delete(-1)
    
    return top_features
    