"""Ensure public package boundaries are not violated by re-exports from _internal."""

import ast
from pathlib import Path

import algo_trading_engine

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PUBLIC_PACKAGES = ("dto", "vo", "enums", "indicators", "plotting", "database")


def _imported_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
    return modules


def test_root_lazy_exports_do_not_target_internal():
    lazy_exports = algo_trading_engine._LAZY_EXPORTS  # type: ignore[attr-defined]
    lazy_paths = {path for path, _ in lazy_exports.values()}
    lazy_paths.update(
        f"algo_trading_engine.{name}" for name in algo_trading_engine._LAZY_SUBMODULES  # type: ignore[attr-defined]
    )
    for path in lazy_paths:
        assert "_internal" not in path, path


def test_public_package_inits_only_import_within_package():
    src = _REPO_ROOT / "src" / "algo_trading_engine"
    for package in _PUBLIC_PACKAGES:
        init_path = src / package / "__init__.py"
        if not init_path.is_file():
            continue
        for module in _imported_modules(init_path.read_text()):
            if not module.startswith("algo_trading_engine."):
                continue
            assert "_internal" not in module, f"{init_path}: {module}"
            suffix = module.removeprefix("algo_trading_engine.")
            if suffix == package or suffix.startswith(f"{package}."):
                continue
            assert False, f"{init_path} imports outside package: {module}"
