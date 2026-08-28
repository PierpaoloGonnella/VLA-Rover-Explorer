from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import Callable
from typing import Any

try:
    from bleak import BleakClient, BleakScanner
except ImportError:  # pragma: no cover - allows simulation in minimal environments
    BleakClient = BleakScanner = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)
UART_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"


def encode_command(command: str) -> bytes:
    return (command.rstrip("\r\n") + "\n").encode("ascii")


def parse_frame(frame: str) -> tuple[str, list[str]] | None:
    parts = frame.strip().split("#")
    if len(parts) < 2 or not parts[0]:
        return None
    return parts[0], [part for part in parts[1:] if part != ""]


class RoverBle:
    def __init__(
        self,
        device_name: str = "BT05",
        characteristic_uuid: str = UART_UUID,
        reconnect_attempts: int = 4,
        backoff_seconds: float = 0.5,
        client_factory: Callable[[Any], Any] | None = None,
        scanner: Any | None = None,
    ):
        self.device_name = device_name
        self.characteristic_uuid = characteristic_uuid.lower()
        self.reconnect_attempts = reconnect_attempts
        self.backoff_seconds = backoff_seconds
        self._client_factory = client_factory or BleakClient
        self._scanner = scanner or BleakScanner
        self._client: Any | None = None
        self._characteristic: Any | None = None
        self._write_with_response = True
        self._write_lock = asyncio.Lock()
        self._battery_mv: int | None = None
        self._battery_received_at: float | None = None
        self._sonar_cm: int | None = None
        self._sonar_left_cm: int | None = None
        self._sonar_right_cm: int | None = None
        self._sonar_scan_sequence = 0
        self._obstacle_blocked = False
        self._notify_buffer = ""
        self._windows_com_prepared = False

    @property
    def battery_mv(self) -> int | None:
        return self._battery_mv

    @property
    def battery_received_at(self) -> float | None:
        return self._battery_received_at

    @property
    def sonar_cm(self) -> int | None:
        return self._sonar_cm

    @property
    def obstacle_blocked(self) -> bool:
        return self._obstacle_blocked

    @property
    def sonar_left_cm(self) -> int | None:
        return self._sonar_left_cm

    @property
    def sonar_right_cm(self) -> int | None:
        return self._sonar_right_cm

    @property
    def sonar_scan_sequence(self) -> int:
        return self._sonar_scan_sequence

    @property
    def connected(self) -> bool:
        return bool(self._client and self._client.is_connected)

    async def connect(self) -> None:
        if self.connected:
            return
        if self._scanner is None or self._client_factory is None:
            raise RuntimeError("bleak is required for real BLE transport")
        self._prepare_windows_console_thread()
        device = await self._scanner.find_device_by_name(self.device_name)
        if device is None:
            raise ConnectionError(f"BLE device {self.device_name!r} not found")
        self._client = self._client_factory(device)
        await self._client.connect()
        await self._discover_characteristic()
        try:
            await self._client.start_notify(self._characteristic, self._notification_handler)
        except Exception:
            LOGGER.warning("Characteristic does not support notifications", exc_info=True)

    def _prepare_windows_console_thread(self) -> None:
        """Undo accidental STA initialization before using WinRT from this CLI.

        OpenCV and optional Windows packages can initialize the console's main
        thread as STA. Bleak requires MTA when there is no GUI message pump.
        """
        if self._windows_com_prepared or sys.platform != "win32" or self._scanner is not BleakScanner:
            return
        try:
            from bleak.backends.winrt.util import uninitialize_sta
        except ImportError:  # pragma: no cover - non-WinRT/older Bleak
            pass
        else:
            uninitialize_sta()
        self._windows_com_prepared = True

    async def _discover_characteristic(self) -> None:
        services = self._client.services
        chars = [char for service in services for char in service.characteristics]
        preferred = next((c for c in chars if c.uuid.lower() == self.characteristic_uuid), None)
        candidates = [
            c for c in chars
            if "notify" in c.properties
            and ({"write", "write-without-response"} & set(c.properties))
        ]
        self._characteristic = preferred or (candidates[0] if candidates else None)
        if self._characteristic is None:
            raise ConnectionError("No BLE characteristic with UART UUID or write+notify properties")
        properties = set(self._characteristic.properties)
        self._write_with_response = "write" in properties

    def _notification_handler(self, _sender: Any, data: bytearray) -> None:
        self._notify_buffer += bytes(data).decode("ascii", errors="ignore")
        while "\n" in self._notify_buffer:
            raw, self._notify_buffer = self._notify_buffer.split("\n", 1)
            self._consume_frame(raw)
        # Several HM-10 clones emit one complete frame per notification without a
        # newline even though outbound UART commands require one.
        if self._notify_buffer.endswith("#"):
            raw, self._notify_buffer = self._notify_buffer, ""
            self._consume_frame(raw)

    def _consume_frame(self, raw: str) -> None:
        parsed = parse_frame(raw)
        if parsed and parsed[0] == "I" and parsed[1]:
            try:
                self._battery_mv = int(parsed[1][0])
                self._battery_received_at = time.monotonic()
            except ValueError:
                LOGGER.debug("Invalid battery frame: %s", raw)
        elif parsed and parsed[0] == "E" and parsed[1]:
            try:
                self._sonar_cm = int(parsed[1][0])
                self._obstacle_blocked = len(parsed[1]) > 1 and int(parsed[1][1]) != 0
                if len(parsed[1]) > 2:
                    self._sonar_left_cm = int(parsed[1][2])
                if len(parsed[1]) > 3:
                    self._sonar_right_cm = int(parsed[1][3])
                if len(parsed[1]) > 4:
                    self._sonar_scan_sequence = int(parsed[1][4])
            except ValueError:
                LOGGER.debug("Invalid ultrasonic frame: %s", raw)

    async def _reconnect(self) -> None:
        for attempt in range(self.reconnect_attempts):
            try:
                await self.connect()
                return
            except Exception:
                if attempt + 1 == self.reconnect_attempts:
                    raise
                await asyncio.sleep(self.backoff_seconds * (2**attempt))

    async def send(self, command: str) -> None:
        payload = encode_command(command)
        async with self._write_lock:
            if not self.connected:
                await self._reconnect()
            try:
                await self._client.write_gatt_char(
                    self._characteristic, payload, response=self._write_with_response
                )
            except Exception:
                LOGGER.warning("BLE write failed; reconnecting", exc_info=True)
                try:
                    await self._client.disconnect()
                except Exception:
                    pass
                self._client = None
                await self._reconnect()
                await self._client.write_gatt_char(
                    self._characteristic, payload, response=self._write_with_response
                )

    async def disconnect(self) -> None:
        """Every connected disconnect path first issues an explicit motor stop."""
        try:
            if self.connected:
                await self.send("A#0#0#")
        finally:
            if self._client is not None:
                try:
                    await self._client.disconnect()
                finally:
                    self._client = None
                    self._characteristic = None


class MockBle:
    """Simulator transport with the same public interface as :class:`RoverBle`."""

    def __init__(self, simulator: Any):
        self.simulator = simulator
        self.commands: list[str] = []
        self._connected = False

    @property
    def battery_mv(self) -> int:
        return self.simulator.battery_mv

    @property
    def sonar_cm(self) -> None:
        return None

    @property
    def obstacle_blocked(self) -> bool:
        return False

    @property
    def sonar_left_cm(self) -> None:
        return None

    @property
    def sonar_right_cm(self) -> None:
        return None

    @property
    def sonar_scan_sequence(self) -> int:
        return 0

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._connected = True

    async def send(self, command: str) -> None:
        if not self._connected:
            await self.connect()
        normalized = command.rstrip("\r\n")
        self.commands.append(normalized)
        await self.simulator.command(normalized)

    async def disconnect(self) -> None:
        try:
            await self.send("A#0#0#")
        finally:
            self._connected = False
