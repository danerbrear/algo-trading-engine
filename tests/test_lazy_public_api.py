"""
Tests for lazy public API loading (PEP 562).

Ensures importing the package or lightweight sub-packages does not eagerly
load backtest engines, data retrievers, or ML dependencies.
"""

import importlib
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHONPATH = str(_REPO_ROOT / "src")


def _run_isolated_import_check(snippet: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": _PYTHONPATH},
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_package_import_does_not_load_backtest_or_sklearn():
    out = _run_isolated_import_check(
        "import sys\n"
        "import algo_trading_engine\n"
        "print('sklearn', 'sklearn' in sys.modules)\n"
        "print('backtest_main', 'algo_trading_engine.backtest.main' in sys.modules)\n"
        "print('data_retriever', 'algo_trading_engine.common.data_retriever' in sys.modules)\n"
    )
    assert "sklearn False" in out
    assert "backtest_main False" in out
    assert "data_retriever False" in out


def test_subpackage_import_does_not_load_backtest_or_sklearn():
    out = _run_isolated_import_check(
        "import sys\n"
        "import algo_trading_engine.dto\n"
        "print('sklearn', 'sklearn' in sys.modules)\n"
        "print('backtest_main', 'algo_trading_engine.backtest.main' in sys.modules)\n"
    )
    assert "sklearn False" in out
    assert "backtest_main False" in out


def test_enums_subpackage_import_does_not_load_backtest_or_sklearn():
    out = _run_isolated_import_check(
        "import sys\n"
        "import algo_trading_engine.enums\n"
        "print('sklearn', 'sklearn' in sys.modules)\n"
        "print('backtest_main', 'algo_trading_engine.backtest.main' in sys.modules)\n"
    )
    assert "sklearn False" in out
    assert "backtest_main False" in out


def test_backtest_config_lazy_load_does_not_load_backtest_main():
    out = _run_isolated_import_check(
        "import sys\n"
        "from algo_trading_engine import BacktestConfig\n"
        "assert BacktestConfig is not None\n"
        "print('backtest_main', 'algo_trading_engine.backtest.main' in sys.modules)\n"
    )
    assert "backtest_main False" in out


def test_backtest_package_config_import_does_not_load_main():
    out = _run_isolated_import_check(
        "import sys\n"
        "from algo_trading_engine.backtest.config import VolumeConfig\n"
        "assert VolumeConfig is not None\n"
        "print('backtest_main', 'algo_trading_engine.backtest.main' in sys.modules)\n"
    )
    assert "backtest_main False" in out


def test_lazy_exports_resolve_and_cache():
    from algo_trading_engine import Strategy, BacktestConfig

    assert Strategy is not None
    assert BacktestConfig is not None

    import algo_trading_engine as pkg

    assert pkg.Strategy is Strategy
    assert pkg.BacktestConfig is BacktestConfig


def test_lazy_backtest_engine_export_resolves():
    from algo_trading_engine import BacktestEngine

    assert BacktestEngine is not None


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
