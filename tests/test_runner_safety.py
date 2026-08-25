import asyncio
import time

import pytest

from rover_explorer.runner import _offload


@pytest.mark.asyncio
async def test_blocking_policy_work_does_not_starve_safety_event_loop():
    safety_task_ran = asyncio.Event()

    async def safety_task():
        await asyncio.sleep(0.01)
        safety_task_ran.set()

    def blocking_vlm_call():
        time.sleep(0.08)
        return "done"

    result, _ = await asyncio.gather(_offload(blocking_vlm_call), safety_task())
    assert result == "done"
    assert safety_task_ran.is_set()
