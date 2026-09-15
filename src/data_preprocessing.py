import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path:str) -> pd.DataFrame:
    df = pd.read_csv("data/raw/Mall_Customers.csv")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    print("Sample Data:")
    print(df.head())

    print("\nDimensions:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\nInfo:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())


def encode_categorical(df: pd.DataFrame, column: str = "Gender") -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])

    return df


def scale_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[features] = scaler.fit_transform(df[features])

    return df_scaled


def preprocess(raw_path: str, features: list, encode_gender: bool = True):
    df = load_data(raw_path)
    inspect_data(df)

    if encode_gender and "Gender" in df.columns:
        df = encode_categorical(df, "Gender")

    df_scaled = scale_features(df, features)

    return df, df_scaled