import pandas as pd


def summarize_anomalies(
    name: str,
    y_test: pd.Series,
    anomalies: pd.Series,
    scores: pd.Series,
    top_n: int = 10,
) -> None:
    """
    Print basic summary and top-N anomalous points for a detector.
    """
    assert (y_test.index == anomalies.index).all()
    assert (y_test.index == scores.index).all()

    n_total = len(y_test)
    n_anom = anomalies.sum()

    print(f"=== {name} ===")
    print(f"Total test points: {n_total}")
    print(f"Flagged anomalies: {n_anom} ({n_anom / n_total:.1%})")

    if n_anom > 0:
        df = pd.DataFrame(
            {
                "value": y_test,
                "score": scores,
                "is_anomaly": anomalies,
            }
        )
        top = df[df["is_anomaly"]].sort_values("score", ascending=False).head(top_n)
        print("\nTop anomalous points:")
        print(top[["value", "score"]])
    print("\n")
