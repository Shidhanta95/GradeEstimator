import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from data_preprocessing import preprocessing
from imblearn.over_sampling import RandomOverSampler

# def SMOTEUpsampling(df):
#     #smote based upsampling 
#     # transform the dataset
#     oversample = RandomOverSampler(sampling_strategy='minority')
#     column = df.columns
#     y = df[column[-1]]
#     x = df.drop('fraud_reported', axis = 1)
#     # x = np.nan_to_num(x)
#     # y = np.nan_to_num(y)
#     x,y = oversample.fit_resample(x,y)
#     x = pd.DataFrame(x)
#     y = pd.DataFrame(y)

#     df_concat = pd.concat([x,y], axis=1) 
#     return df_concat

def featureEngineering():
    df = preprocessing()
    Y_ = df['Class']
    
    Y = pd.DataFrame(Y_)
    print(Y.head())
    X_ = df.drop(['Class'], axis=1)

    # extracting categorical columns
    cat_df = X_.select_dtypes(include = ['object'])
    num_df = X_.select_dtypes(include=["number"])

    lb = LabelEncoder()
    cat_df = cat_df.apply(LabelEncoder().fit_transform)

    scaler = MinMaxScaler()

    scaler.fit(num_df)
    scaled = scaler.fit_transform(num_df)
    scaled_df = pd.DataFrame(scaled, columns=num_df.columns)

    Y = Y.apply(lb.fit_transform)

    X = pd.concat([cat_df,scaled_df,Y], axis = 1)
    print(X.head())
    X.to_csv("data_mod.csv", index = False)
    return X

featureEngineering()
