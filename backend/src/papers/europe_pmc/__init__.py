"""Europe PMC service package.

Re-exports EuropePMCService for backward compatibility.
"""

from backend.src.papers.europe_pmc.service import EuropePMCService
from backend.src.common.sanitizer import sanitize

__all__ = ["EuropePMCService", "sanitize"]
