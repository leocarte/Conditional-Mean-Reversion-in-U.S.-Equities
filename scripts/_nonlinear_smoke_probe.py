"""Exit 0 iff torch and xgboost can share one process and build a DMatrix.

``scripts/run_nonlinear_smoke.sh`` runs this in an isolated subprocess to decide
whether the nonlinear benchmark nonlinear smoke can run strictly (healthy stack, e.g.
Linux / cluster / CI) or must be skipped (e.g. macOS conda, where the duplicate
OpenMP runtime segfaults). Importing both is *not* enough to trigger the crash --
it fires when xgboost first touches numpy data (``xgboost/data.py``
``_meta_from_numpy``) -- so the probe also constructs a small ``DMatrix``.
"""

from __future__ import annotations

import numpy as np
import torch  # noqa: F401  (loads its OpenMP runtime, like the real pipeline)
import xgboost as xgb

xgb.DMatrix(np.zeros((4, 2), dtype="float32"), label=np.zeros(4, dtype="float32"))
print("ok")
