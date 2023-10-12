import pandas as pd

def load_data() :
    df = pd.read_csv(r'data.csv')
    return df




load_data()