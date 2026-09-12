# Phase 1 compat shim (delete in Phase 3). Canonical: backend.src.settings.
# Alias the module object so `from backend.config import _anything` keeps working.
import sys as _sys

from backend.src import settings as _canonical

_sys.modules[__name__] = _canonical
