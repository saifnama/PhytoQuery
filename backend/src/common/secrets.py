"""Signing-secret files: atomic first-write, 0600, never rotated.

Two writers racing on a cold boot must not interleave bytes: the file is
created with ``'xb'`` (O_EXCL — exactly one winner), the loser reads the
winner's bytes. A crash between create and write can leave a 0-byte file;
that is detected and retried once. Permissions are tightened best-effort
(0600 on POSIX; Windows ACLs ignore it, hence the try/except).
"""
import os
import secrets
from pathlib import Path


def _restrict(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def get_or_create_secret(path: str | Path, *, env_var: str, num_bytes: int = 32) -> bytes:
    """Return the secret, creating the file atomically on first use."""
    p = Path(path)
    override = os.getenv(env_var)
    if override:
        return override.encode("utf-8")
    p.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            with open(p, "xb") as handle:
                token = secrets.token_hex(num_bytes).encode("utf-8")
                handle.write(token)
            _restrict(p)
            return token
        except FileExistsError:
            pass
        data = p.read_bytes().strip() if p.exists() else b""
        if data:
            _restrict(p)  # harden pre-existing files too
            return data
        try:
            p.unlink()
        except OSError:
            pass
    raise OSError(f"could not initialize secret file {p}")
