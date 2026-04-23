from src.data_ingestion import load_data
from src.preprocessing import handle_missing, handle_outliers
from src.feature_engineering import add_features
from src.train import train_models


def main():
    df = load_data()
    df = handle_missing(df)
    df = handle_outliers(df)
    df = add_features(df)

    train_models(df)

    print("pipeline done sucessfully")


if __name__ == "__main__":
    main()