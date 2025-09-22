"""ProjectPaths helper centralizing filesystem path resolution.

Avoid ad-hoc relative paths by using this helper. All paths are resolved
relative to the detected project root (directory containing `pyproject.toml`,
`setup.py` or `.git`).
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def project_root() -> Path:
    candidates = ["pyproject.toml", "setup.py", ".git"]
    cwd = Path(__file__).resolve().parent.parent.parent  # src/toaripi_slm
    for parent in [cwd, *cwd.parents]:
        for marker in candidates:
            if (parent / marker).exists():
                return parent
    return cwd


class ProjectPaths:
    _root: Path = project_root()

    @classmethod
    def root(cls) -> Path:
        return cls._root

    @classmethod
    def data_raw(cls) -> Path:
        return cls._root / "data" / "raw"

    @classmethod
    def data_processed(cls) -> Path:
        return cls._root / "data" / "processed"

    @classmethod
    def models_hf(cls) -> Path:
        return cls._root / "models" / "hf"

    @classmethod
    def models_gguf(cls) -> Path:
        return cls._root / "models" / "gguf"

    @classmethod
    def training_runs(cls) -> Path:
        return cls.models_hf() / "training_runs"

    @classmethod
    def configs(cls) -> Path:
        return cls._root / "configs"

    @classmethod
    def checkpoints(cls) -> Path:
        return cls._root / "checkpoints"

    @classmethod
    def ensure_all(cls) -> None:
        for p in [
            cls.data_raw(),
            cls.data_processed(),
            cls.models_hf(),
            cls.models_gguf(),
            cls.checkpoints(),
            cls.training_runs(),
        ]:
            p.mkdir(parents=True, exist_ok=True)
