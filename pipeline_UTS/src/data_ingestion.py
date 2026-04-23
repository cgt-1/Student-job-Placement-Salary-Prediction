import pandas as pd

def load_data():
    features_df = pd.read_csv('data/A.csv')
    targets_df = pd.read_csv('data/A_targets.csv')
    df = pd.merge(features_df, targets_df, on='Student_ID')
    return df