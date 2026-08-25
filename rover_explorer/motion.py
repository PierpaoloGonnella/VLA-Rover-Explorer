from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Protocol


class Action(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    ARC_LEFT = "arc_left"
    ARC_RIGHT = "arc_right"
    STOP = "stop"


class CommandSink(Protocol):
    async def send(self, command: str) -> None: ...


def motor_command(action: Action, speed: int) -> str:
    speed = max(0, min(255, int(speed)))
    commands = {
        Action.FORWARD: (speed, speed),
        Action.BACKWARD: (-speed, -speed),
        Action.TURN_LEFT: (-speed, speed),
        Action.TURN_RIGHT: (speed, -speed),
        Action.ARC_LEFT: (speed // 2, speed),
        Action.ARC_RIGHT: (speed, speed // 2),
        Action.STOP: (0, 0),
    }
    left, right = commands[action]
    return f"A#{left}#{right}#"


async def pulse(
    ble: CommandSink,
    action: Action,
    speed: int,
    duration_ms: int,
    settle_ms: int = 250,
) -> None:
    """Execute exactly one bounded motor pulse, with stop guaranteed on cancellation."""
    try:
        await ble.send(motor_command(action, speed))
        await asyncio.sleep(max(0, duration_ms) / 1000)
    finally:
        await asyncio.shield(ble.send("A#0#0#"))
    await asyncio.sleep(max(0, settle_ms) / 1000)


class MotionWatchdog:
    """Independent fail-safe which stops motors if completed pulses cease."""

    def __init__(self, ble: CommandSink, timeout: float = 2.0, poll_interval: float = 0.1):
        self.ble = ble
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._last_completed = time.monotonic()
        self._task: asyncio.Task[None] | None = None
        self._stopped = False

    def pulse_completed(self) -> None:
        self._last_completed = time.monotonic()

    async def _run(self) -> None:
        while not self._stopped:
            await asyncio.sleep(self.poll_interval)
            if time.monotonic() - self._last_completed > self.timeout:
                await self.ble.send("A#0#0#")
                self._last_completed = time.monotonic()

    async def start(self) -> None:
        if self._task is None:
            self._stopped = False
            self._task = asyncio.create_task(self._run(), name="motor-watchdog")

    async def stop(self) -> None:
        self._stopped = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self.ble.send("A#0#0#")
