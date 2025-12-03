from pathlib import Path

import pandas as pd

from time_series_transformer.data_pipeline.preprocessing import load_csv_to_df
from time_series_transformer.data_pipeline.labels import (
    load_label_times,
    make_point_labels_from_times,
)
from time_series_transformer.split import train_test_split_series
from time_series_transformer.evaluation import summarize_anomalies
from time_series_transformer.models.baseline.arima import ARIMAResidualAnomalyDetector
from time_series_transformer.models.baseline.isolation_forest import (
    IsolationForestAnomalyDetector,
)
from time_series_transformer.models.baseline.rolling_zscore import (
    RollingZScoreAnomalyDetector,
)
from time_series_transformer.models.baseline.lstm import LSTMForecastAnomalyDetector

from time_series_transformer.config import (
    DATA_DIR,
    RAW_DATA_DIR,
    TRAIN_RATIO,
    ROLLING_WINDOW,
    ROLLING_Z_THRESH,
    ARIMA_ORDER,
    ARIMA_Z_THRESH,
    ISO_CONTAMINATION,
    RANDOM_STATE,
    LSTM_LOOKBACK,
    LSTM_LR,
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_ERROR_QUANTILE,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
)


LABELS_JSON = DATA_DIR / "labels" / "nab" / "realKnownCause.json"
NYC_TAXI_CSV = (
    RAW_DATA_DIR / "nab" / "realKnownCause" / "realKnownCause" / "nyc_taxi.csv"
)


def main():
    # 1) Load df and put timestamp on the index
    df = load_csv_to_df(NYC_TAXI_CSV, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # y now has a DatetimeIndex
    y = df["value"]

    # 2) Load label times & make pointwise labels with DatetimeIndex
    label_times = load_label_times(LABELS_JSON, "realKnownCause/nyc_taxi.csv")

    # make_point_labels_from_times expects a timestamp column, so give it a df with that column:
    df_for_labels = df.reset_index()  # has 'timestamp' column again
    y_true_all = make_point_labels_from_times(
        df_for_labels, label_times, timestamp_col="timestamp"
    )

    # IMPORTANT: align index with y (DatetimeIndex)
    y_true_all.index = y.index

    # 3) Time-based split by position (index stays DatetimeIndex)
    y_train, y_test = train_test_split_series(y, train_ratio=TRAIN_RATIO)
    y_true_train, y_true_test = train_test_split_series(
        y_true_all, train_ratio=TRAIN_RATIO
    )

    # 4) Models
    models = {
        # "Rolling Z-Score": RollingZScoreAnomalyDetector(
        #    window=ROLLING_WINDOW, z_thresh=ROLLING_Z_THRESH
        # ),
        "ARIMA Residual": ARIMAResidualAnomalyDetector(
            order=ARIMA_ORDER, z_thresh=ARIMA_Z_THRESH
        ),
        "Isolation Forest": IsolationForestAnomalyDetector(
            contamination=ISO_CONTAMINATION, random_state=RANDOM_STATE
        ),
        "LSTM Forecast": LSTMForecastAnomalyDetector(
            lookback=LSTM_LOOKBACK,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT,
            batch_size=LSTM_BATCH_SIZE,
            lr=LSTM_LR,
            epochs=LSTM_EPOCHS,
            error_quantile=LSTM_ERROR_QUANTILE,
            device="cpu",
        ),
    }

    # 5) Train + evaluate
    for name, model in models.items():
        model.fit(y_train)
        scores = model.decision_function(y_test)  # index = DatetimeIndex
        anomalies = model.predict(y_test)  # index = DatetimeIndex

        summarize_anomalies(
            name=name,
            y_test=y_test,
            anomalies=anomalies,
            scores=scores,
            y_true_labels=y_true_test,  # also DatetimeIndex
            top_n=10,
        )


if __name__ == "__main__":
    main()
