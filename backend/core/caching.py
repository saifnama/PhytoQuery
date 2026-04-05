import os
import json
import time
import hashlib
import gzip
import glob as _glob
from typing import Optional, Any

DEFAULT_TTL_SECONDS = 3600  # 1 hour
MAX_CACHE_FILES = 500
CACHE_VERSION = "v2"  # Increment when cache format changes


class SimpleCache:
    """File-based cache with hash-prefix subdirectories (git-style).

    Features:
    - Gzip compression (70-90% smaller)
    - Cache versioning (auto-invalidate on code changes)
    - TTL expiration
    - Max file limit with LRU eviction

    Structure:
        cache/europepmc/
        ├── 31/
        │   └── 31d30ca7888979622df3d4625472e1db.json.gz
        ├── 71/
        │   └── 71cf8c0aaab59e28550465d7db19d454.json.gz
    """

    def __init__(
        self,
        cache_dir: str,
        ttl: int = DEFAULT_TTL_SECONDS,
        max_files: int = MAX_CACHE_FILES,
    ):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.max_files = max_files
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_key(self, identifier: str) -> str:
        return hashlib.sha256(identifier.encode()).hexdigest()

    def _get_path(self, key: str) -> str:
        """Return file path with hash-prefix subdirectory (first 2 chars)."""
        prefix = key[:2]
        subdir = os.path.join(self.cache_dir, prefix)
        os.makedirs(subdir, exist_ok=True)
        return os.path.join(subdir, f"{key}.json.gz")

    def get(self, identifier: str) -> Optional[Any]:
        key = self._get_key(identifier)
        cache_file = self._get_path(key)

        # Try new compressed format first
        if os.path.exists(cache_file):
            try:
                with gzip.open(cache_file, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                if self._is_valid(data):
                    return data
                # Invalid/expired - remove
                os.remove(cache_file)
                self._cleanup_empty_dirs(os.path.dirname(cache_file))
            except Exception:
                pass

        # Try legacy .json format (backward compatibility)
        legacy_path = cache_file.replace(".json.gz", ".json")
        if os.path.exists(legacy_path):
            try:
                with open(legacy_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if self._is_valid(data):
                    # Migrate to new format
                    self.set(identifier, data)
                    os.remove(legacy_path)
                    self._cleanup_empty_dirs(os.path.dirname(legacy_path))
                    return data
            except Exception:
                pass

        return None

    def _is_valid(self, data: dict) -> bool:
        """Check if cache entry is valid (version + TTL)."""
        # Version check
        if data.get("_version") != CACHE_VERSION:
            return False
        # TTL check
        created = data.get("_cached_at", 0)
        if created and (time.time() - created) > self.ttl:
            return False
        return True

    def set(self, identifier: str, data: Any):
        # Evict oldest files if cache is too large
        self._evict_if_needed()

        key = self._get_key(identifier)
        cache_file = self._get_path(key)

        # Add metadata
        cache_data = {
            **data,
            "_version": CACHE_VERSION,
            "_cached_at": time.time(),
        }

        try:
            with gzip.open(cache_file, "wt", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass

    def _evict_if_needed(self):
        """Remove oldest cache files if count exceeds max_files."""
        files = sorted(
            _glob.glob(os.path.join(self.cache_dir, "**", "*.json*"), recursive=True),
            key=os.path.getmtime,
        )
        while len(files) >= self.max_files:
            oldest = files.pop(0)
            try:
                os.remove(oldest)
                self._cleanup_empty_dirs(os.path.dirname(oldest))
            except OSError:
                break

    def _cleanup_empty_dirs(self, dir_path: str):
        """Remove empty directories up to cache_dir root."""
        while dir_path != self.cache_dir and dir_path.startswith(self.cache_dir):
            try:
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    dir_path = os.path.dirname(dir_path)
                else:
                    break
            except OSError:
                break

    def clear(self):
        for root, dirs, files in os.walk(self.cache_dir, topdown=False):
            for f in files:
                if f.endswith(".json") or f.endswith(".json.gz"):
                    os.remove(os.path.join(root, f))
            for d in dirs:
                dir_path = os.path.join(root, d)
                try:
                    os.rmdir(dir_path)
                except OSError:
                    pass


# Initialize caches
_data_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"
)
pmc_cache = SimpleCache(
    os.path.join(_data_dir, "cache", "europepmc"),
    ttl=86400,  # 24 hours — paper content doesn't change often
    max_files=500,
)
ner_cache = SimpleCache(
    os.path.join(_data_dir, "cache", "ner"),
    ttl=86400,  # 24 hours
    max_files=500,
)
doi_cache = SimpleCache(
    os.path.join(_data_dir, "cache", "doi"),
    ttl=86400,  # 24 hours
    max_files=500,
)
