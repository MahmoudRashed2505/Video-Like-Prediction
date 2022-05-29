import pandas as pd
from sklearn.preprocessing import LabelEncoder


def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


def clean_data(df):
    data = df
    data.drop(data[data['video_error_or_removed']==True].index, inplace = True) # Drop the videos that has error or removed
    # Columns that we Won't Need in the Future of Our Project
    un_necessary_columns = ['video_id','comments_disabled','ratings_disabled','trending_date',
                            'publish_time','video_error_or_removed','channel_title','video_description','title','tags']
    data.drop(un_necessary_columns,axis = 1, inplace = True) # Drop the un-necessary columns

    
    return data