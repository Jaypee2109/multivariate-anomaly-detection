from time_series_transformer.analysis.eda import run_anomaly_eda_from_artifacts

artifacts_path = "artifacts/anomalies/baseline_anomalies.csv"

if __name__ == "__main__":
    anomaly_layout = run_anomaly_eda_from_artifacts(
        artifacts_path=artifacts_path,
        timestamp_col="timestamp",
        value_col="value",
        html_name="timeseries_anomalies.html",
    )
