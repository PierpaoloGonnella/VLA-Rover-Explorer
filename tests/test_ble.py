import pytest

from rover_explorer.ble import MockBle, RoverBle, encode_command, parse_frame


class FakeCharacteristic:
    uuid = "0000ffe1-0000-1000-8000-00805f9b34fb"
    properties = ["write", "notify"]


class FakeService:
    characteristics = [FakeCharacteristic()]


class FakeScanner:
    async def find_device_by_name(self, name):
        return object()


class FakeClient:
    def __init__(self, device):
        self.is_connected = False
        self.services = [FakeService()]
        self.writes = []

    async def connect(self):
        self.is_connected = True

    async def start_notify(self, characteristic, callback):
        self.callback = callback

    async def write_gatt_char(self, characteristic, data, response):
        self.writes.append((bytes(data), response))

    async def disconnect(self):
        self.is_connected = False


def test_protocol_encoding_and_parsing():
    assert encode_command("A#150#150#") == b"A#150#150#\n"
    assert encode_command("A#0#0#\n") == b"A#0#0#\n"
    assert parse_frame("I#7390#") == ("I", ["7390"])
    assert parse_frame("E#24#1#40#75#3#") == ("E", ["24", "1", "40", "75", "3"])


@pytest.mark.asyncio
async def test_disconnect_always_stops_and_battery_notification_is_parsed():
    ble = RoverBle(client_factory=FakeClient, scanner=FakeScanner())
    await ble.connect()
    client = ble._client
    client.callback(None, bytearray(b"I#7420#\n"))
    assert ble.battery_mv == 7420
    client.callback(None, bytearray(b"E#24#1#40#75#3#\n"))
    assert ble.sonar_cm == 24
    assert ble.obstacle_blocked is True
    assert ble.sonar_left_cm == 40
    assert ble.sonar_right_cm == 75
    assert ble.sonar_scan_sequence == 3
    client.callback(None, bytearray(b"E#42#0#\n"))
    assert ble.sonar_cm == 42
    assert ble.obstacle_blocked is False
    await ble.disconnect()
    assert client.writes[-1][0] == b"A#0#0#\n"


class TinySimulator:
    battery_mv = 7000

    def __init__(self):
        self.commands = []

    async def command(self, command):
        self.commands.append(command)


@pytest.mark.asyncio
async def test_mock_disconnect_always_stops():
    simulator = TinySimulator()
    ble = MockBle(simulator)
    await ble.connect()
    await ble.disconnect()
    assert simulator.commands[-1] == "A#0#0#"
