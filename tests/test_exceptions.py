"""Tests for custom exception hierarchy."""

from __future__ import annotations

import pytest

from time_series_transformer.exceptions import (
    ConfigurationError,
    DataNotFoundError,
    DataValidationError,
    ModelNotFittedError,
    TransformerError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_transformer_error(self):
        for exc_cls in (
            DataNotFoundError,
            DataValidationError,
            ModelNotFittedError,
            ConfigurationError,
        ):
            assert issubclass(exc_cls, TransformerError)

    def test_data_not_found_is_file_not_found(self):
        assert issubclass(DataNotFoundError, FileNotFoundError)

    def test_data_validation_is_value_error(self):
        assert issubclass(DataValidationError, ValueError)

    def test_model_not_fitted_is_runtime_error(self):
        assert issubclass(ModelNotFittedError, RuntimeError)

    def test_configuration_error_is_value_error(self):
        assert issubclass(ConfigurationError, ValueError)

    def test_catch_all_with_transformer_error(self):
        with pytest.raises(TransformerError):
            raise DataNotFoundError("test")

        with pytest.raises(TransformerError):
            raise ModelNotFittedError("test")

    def test_catch_with_stdlib_exception(self):
        with pytest.raises(FileNotFoundError):
            raise DataNotFoundError("missing")

        with pytest.raises(RuntimeError):
            raise ModelNotFittedError("not fitted")
