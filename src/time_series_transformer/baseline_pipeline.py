from pathlib import Path
from time_series_transformer.config import (
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
from time_series_transformer.data_pipeline.data_loading import load_timeseries
from time_series_transformer.split import train_test_split_series
from time_series_transformer.evaluation import summarize_anomalies
from time_series_transformer.models.baseline.arima import ARIMAResidualAnomalyDetector
from time_series_transformer.models.baseline.rolling_zscore import (
    RollingZScoreAnomalyDetector,
)
from time_series_transformer.models.baseline.isolation_forest import (
    IsolationForestAnomalyDetector,
)
from time_series_transformer.models.baseline.lstm import LSTMForecastAnomalyDetector
from time_series_transformer.utils.anomaly_io import save_anomaly_artifacts


def run_pipeline(csv_path) -> None:
    # 1. Load data
    y = load_timeseries(csv_path)

    # 2. Train/test split
    y_train, y_test = train_test_split_series(y, train_ratio=TRAIN_RATIO)

    # 3. Define models
    models = {
        # "Rolling Z-Score": RollingZScoreAnomalyDetector(
        #    window=ROLLING_WINDOW,
        #    z_thresh=ROLLING_Z_THRESH,
        # ),
        "ARIMA Residual": ARIMAResidualAnomalyDetector(
            order=ARIMA_ORDER,
            z_thresh=ARIMA_Z_THRESH,
        ),
        "Isolation Forest": IsolationForestAnomalyDetector(
            contamination=ISO_CONTAMINATION,
            random_state=RANDOM_STATE,
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

    scores_dict = {}
    anomalies_dict = {}

    # 4. Train and evaluate
    for name, model in models.items():
        model.fit(y_train)
        scores = model.decision_function(y_test)
        anomalies = model.predict(y_test)

        scores_dict[name] = scores
        anomalies_dict[name] = anomalies

        summarize_anomalies(name, y_test, anomalies, scores)

    # 5. Save artifacts
    artifacts_path = Path("artifacts/anomalies/baseline_anomalies.csv")
    save_anomaly_artifacts(
        y_test=y_test,
        scores_dict=scores_dict,
        anomalies_dict=anomalies_dict,
        out_path=artifacts_path,
    )
