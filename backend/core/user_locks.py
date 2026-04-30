import asyncio
from contextlib import asynccontextmanager


class UserLockManager:
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _get_lock(self, user_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(user_id, asyncio.Lock())

    @asynccontextmanager
    async def lock(self, user_id: str):
        lock = await self._get_lock(user_id)
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()


user_lock_manager = UserLockManager()
