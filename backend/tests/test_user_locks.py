import asyncio

from backend.core.user_locks import UserLockManager


async def _acquire_and_hold(lock_manager, user_id, order):
    async with lock_manager.lock(user_id):
        order.append((user_id, "entered"))
        await asyncio.sleep(0.05)
        order.append((user_id, "leaving"))


def test_same_user_locks_serialize():
    manager = UserLockManager()
    order = []

    async def run_test():
        await asyncio.gather(
            _acquire_and_hold(manager, "sess_a", order),
            _acquire_and_hold(manager, "sess_a", order),
        )

    asyncio.run(run_test())

    assert order == [
        ("sess_a", "entered"),
        ("sess_a", "leaving"),
        ("sess_a", "entered"),
        ("sess_a", "leaving"),
    ]


def test_different_users_do_not_block_each_other():
    manager = UserLockManager()
    order = []

    async def run_test():
        await asyncio.gather(
            _acquire_and_hold(manager, "sess_a", order),
            _acquire_and_hold(manager, "sess_b", order),
        )

    asyncio.run(run_test())

    entered = [item for item in order if item[1] == "entered"]
    assert {item[0] for item in entered} == {"sess_a", "sess_b"}
