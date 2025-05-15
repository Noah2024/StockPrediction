import pandas as pd
import numpy as np

def normalizeAndUpdate(df: pd.DataFrame, *, normParams: dict = None):
    """
    Normalize numeric columns in the DataFrame using min-max normalization.
    
    Args:
        df (pd.DataFrame): The DataFrame to normalize in-place.
        normParams (dict, optional): If provided, uses these normalization
            parameters in the form {column_name: (min, max)}.
    
    Returns:
        dict or None: Computed normalization parameters if normParams not provided;
                      otherwise, returns None.
    """
    computed_params = {}

    for column in df.select_dtypes(include=["number"]).columns:
        if normParams is not None and column in normParams:
            col_min, col_max = normParams[column]
        else:
            col_min = df[column].min()
            col_max = df[column].max()
            computed_params[column] = (col_min, col_max)

        # Avoid divide-by-zero in case of constant column
        if col_max != col_min:
            df[column] = (df[column] - col_min) / (col_max - col_min)
        else:
            df[column] = 0.0

    return None if normParams is not None else computed_params

def denormalize(df: pd.DataFrame, normalization_params: dict):#ChaptGpt Gen
    """
    Reverse min-max normalization using stored min and max values.
    
    Parameters:
        - df: The DataFrame with normalized values
        - normalization_params: Dictionary with column -> (min, max) for each column
    
    Returns:
        - denormalized_df (pd.DataFrame): The denormalized DataFrame
    """
    df = df.copy()

    for column, (col_min, col_max) in normalization_params.items():
        print(column)
        print(type(column))
        df[column] = df[column] * (col_max - col_min) + col_min
    
    return df

def denormalizeNp(array: np.ndarray, normalization_params: dict):
    """
    Reverse min-max normalization for a numpy array using the order of keys in normalization_params.

    Parameters:
        - array: The normalized numpy array (shape: [n_samples, n_features])
        - normalization_params: Dictionary with column -> (min, max) for each column

    Returns:
        - denormalized_array (np.ndarray): The denormalized numpy array
    """
    denorm_array = array.copy()
    keys = list(normalization_params.keys())
    for idx, col in enumerate(keys):
        col_min, col_max = normalization_params[col]
        denorm_array[:, idx] = denorm_array[:, idx] * (col_max - col_min) + col_min
    return denorm_array

def convertTsToEpoch(df):#ChatGPT generated
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp"] = df["timestamp"].astype("int64") // 1_000_000_000