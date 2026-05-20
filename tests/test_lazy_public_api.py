"""
Tests for lazy public API loading (PEP 562).

Ensures importing the package or lightweight sub-packages does not eagerly
load backtest engines, data retrievers, or ML dependencies.
"""

import importlib
import sys

import pytest


def _unload_algo_trading_engine() -> None:
    """Remove algo_trading_engine and submodules from sys.modules for a clean import."""
    to_remove = [key for key in sys.modules if key == "algo_trading_engine" or key.startswith("algo_trading_engine.")]
    for key in to_remove:
        del sys.modules[key]


@pytest.fixture(autouse=True)
def clean_package_imports():
    _unload_algo_trading_engine()
    yield
    _unload_algo_trading_engine()


def test_package_import_does_not_load_backtest_or_sklearn():
    import algo_trading_engine  # noqa: F401

    assert "sklearn" not in sys.modules
    assert "algo_trading_engine.backtest.main" not in sys.modules
    assert "algo_trading_engine.common.data_retriever" not in sys.modules


def test_subpackage_import_does_not_load_backtest_or_sklearn():
    import algo_trading_engine.dto  # noqa: F401

    assert "sklearn" not in sys.modules
    assert "algo_trading_engine.backtest.main" not in sys.modules


def test_enums_subpackage_import_does_not_load_backtest_or_sklearn():
    import algo_trading_engine.enums  # noqa: F401

    assert "sklearn" not in sys.modules
    assert "algo_trading_engine.backtest.main" not in sys.modules


def test_backtest_config_lazy_load_does_not_load_backtest_main():
    from algo_trading_engine import BacktestConfig

    assert BacktestConfig is not None
    assert "algo_trading_engine.backtest.main" not in sys.modules


def test_backtest_package_config_import_does_not_load_main():
    from algo_trading_engine.backtest.config import VolumeConfig

    assert VolumeConfig is not None
    assert "algo_trading_engine.backtest.main" not in sys.modules


def test_lazy_exports_resolve_and_cache():
    from algo_trading_engine import BacktestEngine, Strategy, BacktestConfig

    assert BacktestEngine is not None
    assert Strategy is not None
    assert BacktestConfig is not None

    import algo_trading_engine as pkg

    assert pkg.BacktestEngine is BacktestEngine
    assert pkg.Strategy is Strategy


def test_lazy_submodule_exports_resolve():
    from algo_trading_engine import vo, enums

    assert vo is importlib.import_module("algo_trading_engine.vo")
    assert enums is importlib.import_module("algo_trading_engine.enums")

    import algo_trading_engine as pkg

    assert pkg.vo is vo
    assert pkg.enums is enums


def test_dir_includes_public_api_names():
    import algo_trading_engine as pkg

    names = pkg.__dir__()
    assert "BacktestEngine" in names
    assert "dto" in names
    assert "enums" in names
