"""Tests for Custom Transformer multivariate anomaly detector."""

from __future__ import annotations

import pandas as pd
import torch


class TestCustomTransformerForecasterModel:
    def test_forward_shape(self):
        from time_series_transformer.models.multivariate.custom_transformer import (
            CustomTransformerForecaster,
        )

        n_features = 5
        model = CustomTransformerForecaster(
            n_features=n_features,
            t2v_dim=8,
            model_dim=16,
            num_heads=2,
            num_layers=1,
        )
        src = torch.randn(4, 10, n_features)  # (batch=4, seq_len=10, features=5)
        offsets = torch.tensor([0, 60, 120, 1440], dtype=torch.long)
        out = model(src, offsets)
        assert out.shape == (4, n_features)

    def test_derive_time_features(self):
        from time_series_transformer.models.multivariate.custom_transformer import (
            CustomTransformerForecaster,
        )

        tf = CustomTransformerForecaster._derive_time_features(torch.tensor([0, 59, 60, 1439]))
        assert tf.shape == (4, 2)
        # minute 0 → minute_of_hour=0.0
        assert tf[0, 0].item() == 0.0
        # minute 59 → minute_of_hour=1.0
        assert abs(tf[1, 0].item() - 1.0) < 1e-6
        # minute 60 → minute_of_hour=0.0, hour_of_day=1/23
        assert tf[2, 0].item() == 0.0
        assert abs(tf[2, 1].item() - 1.0 / 23.0) < 1e-6
        # minute 1439 → minute_of_hour=1.0, hour_of_day=1.0
        assert abs(tf[3, 0].item() - 1.0) < 1e-6
        assert abs(tf[3, 1].item() - 1.0) < 1e-6


class TestCustomTransformerDetector:
    def test_fit_predict(self, sample_multivariate_df):
        from time_series_transformer.models.multivariate.custom_transformer import (
            CustomTransformerDetector,
        )

        det = CustomTransformerDetector(
            lookback=10,
            t2v_dim=8,
            model_dim=16,
            num_heads=2,
            num_layers=1,
            dim_feedforward=32,
            epochs=2,
            batch_size=16,
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
        from time_series_transformer.models.multivariate.custom_transformer import (
            CustomTransformerDetector,
        )

        det = CustomTransformerDetector(
            lookback=10,
            t2v_dim=8,
            model_dim=16,
            num_heads=2,
            num_layers=1,
            dim_feedforward=32,
            epochs=2,
            batch_size=16,
        )
        det.fit(sample_multivariate_df.iloc[:70])

        path = tmp_path / "custom_tf.pt"
        det.save_checkpoint(path)

        loaded = CustomTransformerDetector.load_checkpoint(path)
        test = sample_multivariate_df.iloc[70:]
        scores_orig = det.decision_function(test)
        scores_loaded = loaded.decision_function(test)
        assert len(scores_loaded) == len(test)
        pd.testing.assert_series_equal(scores_orig, scores_loaded)
