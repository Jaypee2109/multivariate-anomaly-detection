import pandas as pd


def train_test_split_series(y: pd.Series, train_ratio: float = 0.7) -> tuple[pd.Series, pd.Series]:
    """
    Time-ordered train/test split for a Series.
    """
    n = len(y)
    n_train = int(n * train_ratio)
    y_train = y.iloc[:n_train]
    y_test = y.iloc[n_train:]
    return y_train, y_test
