"""Europe PMC service package.

Re-exports EuropePMCService for backward compatibility.
"""

from backend.services.europe_pmc.service import EuropePMCService
from backend.core.sanitizer import sanitize

__all__ = ["EuropePMCService", "sanitize"]
