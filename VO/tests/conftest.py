"""Import bootstrap for the test suite.

The package lives under ``src/`` and the runnable scripts under ``tools/`` are
imported as a package from the repository root. Neither is importable from a
bare checkout unless both are on ``sys.path``, and requiring ``pip install -e .``
before ``pytest`` would make a clean clone fail in a way that looks like a code
error rather than a setup step.

Individual modules under ``tools/`` carry the same bootstrap, because each is
also runnable directly as ``python tools/<name>.py``. This file covers the case
where pytest imports them without any of that having run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for _entry in (ROOT / "src", ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))
