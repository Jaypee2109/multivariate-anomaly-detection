"""Tests for config env-var helpers."""

from __future__ import annotations

from time_series_transformer.config import _env_bool, _env_float, _env_int


class TestEnvInt:
    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_INT_VAR", raising=False)
        assert _env_int("TEST_INT_VAR", 42) == 42

    def test_reads_valid_int(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "7")
        assert _env_int("TEST_INT_VAR", 42) == 7

    def test_falls_back_on_invalid(self, monkeypatch):
        monkeypatch.setenv("TEST_INT_VAR", "not_a_number")
        assert _env_int("TEST_INT_VAR", 42) == 42


class TestEnvFloat:
    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_FLOAT_VAR", raising=False)
        assert _env_float("TEST_FLOAT_VAR", 3.14) == 3.14

    def test_reads_valid_float(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "2.718")
        assert _env_float("TEST_FLOAT_VAR", 3.14) == 2.718

    def test_falls_back_on_invalid(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT_VAR", "xyz")
        assert _env_float("TEST_FLOAT_VAR", 3.14) == 3.14


class TestEnvBool:
    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("TEST_BOOL_VAR", raising=False)
        assert _env_bool("TEST_BOOL_VAR", False) is False

    def test_true_values(self, monkeypatch):
        for val in ("true", "1", "yes", "on", "True", "YES"):
            monkeypatch.setenv("TEST_BOOL_VAR", val)
            assert _env_bool("TEST_BOOL_VAR", False) is True

    def test_false_values(self, monkeypatch):
        for val in ("false", "0", "no", "off"):
            monkeypatch.setenv("TEST_BOOL_VAR", val)
            assert _env_bool("TEST_BOOL_VAR", True) is False
