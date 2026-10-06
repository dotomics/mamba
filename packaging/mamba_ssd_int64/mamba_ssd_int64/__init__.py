"""Loader-only copy of mamba-ssm's Triton SSD kernels with the int32-overflow fix.

The files next to this module are byte-identical to ``mamba_ssm/ops/triton/`` on
the dotomics/mamba ``ssd-int64`` branch: upstream main plus state-spaces/mamba#1037
(stack #1038, #1034-#1037) and the layernorm_gated / k_activations hunks of #1016.
They keep upstream's absolute ``mamba_ssm.ops.triton.*`` imports, so importing one
directly would bind the *stock* kernels of the installed mamba-ssm. Importing
``mamba_ssd_int64.<file>`` therefore raises; a loader must execute the files from
``path(name)`` with ``sys.modules`` pointing those imports at each other (see
BOTANIC's ``backbones/mamba2_int64.py``).
"""

from __future__ import annotations

import importlib.abc
import sys
from pathlib import Path

# Dependency order: each file imports only files listed before it.
FILES = (
    "ssd_bmm",
    "ssd_chunk_state",
    "ssd_state_passing",
    "ssd_chunk_scan",
    "layernorm_gated",
    "k_activations",
    "ssd_combined",
)
_DIR = Path(__file__).parent


def path(name: str) -> Path:
    """Path of the kernel file ``name`` (one of ``FILES``)."""
    if name not in FILES:
        raise KeyError(f"{name!r} is not one of {FILES}")
    return _DIR / f"{name}.py"


class _LoaderOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] == __name__ and fullname.rpartition(".")[2] in FILES:
            raise ImportError(
                f"{fullname} is loader-only: imported directly, its mamba_ssm.ops.triton "
                "imports would bind the stock kernels. Load the files from "
                "mamba_ssd_int64.path(name) with a sys.modules loader instead."
            )
        return None


if not any(isinstance(f, _LoaderOnly) for f in sys.meta_path):
    sys.meta_path.insert(0, _LoaderOnly())
