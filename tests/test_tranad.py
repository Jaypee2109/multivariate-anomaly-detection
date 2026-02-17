"""Tests for TranAD multivariate anomaly detector."""

from __future__ import annotations

import pandas as pd
import pytest
import torch


class TestTranADModel:
    def test_forward_shape(self):
        from time_series_transformer.models.multivariate.tranad import TranADModel

        n_features = 4
        model = TranADModel(
            n_features=n_features,
            n_window=5,
            n_heads=4,
            dim_feedforward=8,
        )
        # seq-first: (seq_len, batch, n_features)
        src = torch.randn(5, 3, n_features)
        tgt = torch.randn(1, 3, n_features)

        x1, x2 = model(src, tgt)

        assert x1.shape == (1, 3, n_features)
        assert x2.shape == (1, 3, n_features)

    def test_output_in_zero_one(self):
        """Sigmoid output layer should keep values in [0, 1]."""
        from time_series_transformer.models.multivariate.tranad import TranADModel

        model = TranADModel(n_features=4, n_window=5, n_heads=4)
        src = torch.randn(5, 2, 4)
        tgt = torch.randn(1, 2, 4)
        x1, x2 = model(src, tgt)
        assert (x1 >= 0).all() and (x1 <= 1).all()
        assert (x2 >= 0).all() and (x2 <= 1).all()


class TestTranADAnomalyDetector:
    def test_fit_predict(self, sample_multivariate_df):
        from time_series_transformer.models.multivariate.tranad import (
            TranADAnomalyDetector,
        )

        det = TranADAnomalyDetector(
            lookback=5, n_heads=5, dim_feedforward=8, epochs=2, batch_size=16,
        )
        train = sample_multivariate_df.iloc[:70]
        test = sample_multivariate_df.iloc[70:]
        det.fit(train)

        scores = det.decision_function(test)
        preds = det.predict(test)

        assert isinstance(scores, pd.Series)
        assert len(scores) == len(test)
        assert isinstance(preds, pd.Series)
        assert preds.dtype == bool

    def test_checkpoint_roundtrip(self, tmp_path, sample_multivariate_df):
        from time_series_transformer.models.multivariate.tranad import (
            TranADAnomalyDetector,
        )

        det = TranADAnomalyDetector(
            lookback=5, n_heads=5, dim_feedforward=8, epochs=2, batch_size=16,
        )
        det.fit(sample_multivariate_df.iloc[:70])

        path = tmp_path / "tranad.pt"
        det.save_checkpoint(path)

        loaded = TranADAnomalyDetector.load_checkpoint(path)
        test = sample_multivariate_df.iloc[70:]
        scores_orig = det.decision_function(test)
        scores_loaded = loaded.decision_function(test)
        assert len(scores_loaded) == len(test)
        pd.testing.assert_series_equal(scores_orig, scores_loaded)
