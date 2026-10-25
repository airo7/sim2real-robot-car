"""BLE link to the car's HM-10-style UART module (advertises as "HC-05").

Writes CMD packets to characteristic FFE1 and parses SENS packets from its notifications.
Used by run_hardware.py --link ble [--address <uuid>].
"""

import asyncio
import threading

from bleak import BleakClient, BleakScanner

from hardware.protocol import SensReader, encode_cmd
from hardware.ir_calib import IRCalib, AutoIRCalib

UART_CHAR = "0000ffe1-0000-1000-8000-00805f9b34fb"
DEFAULT_NAME = "HC-05"


def discover(name: str = DEFAULT_NAME, timeout: float = 20.0) -> str | None:
    """Blocking scan; address of the first device whose name contains `name`, or None."""
    async def _scan():
        devs = await BleakScanner.discover(timeout=timeout, return_adv=True)
        for addr, (d, adv) in devs.items():
            dn = d.name or adv.local_name or ""
            if name.lower() in dn.lower():
                return addr
        return None
    return asyncio.run(_scan())


class BleLink:
    """bleak client on a private event loop behind the synchronous SerialLink API."""

    def __init__(self, address: str, char: str = UART_CHAR, connect_timeout: float = 20.0,
                 calib=None):
        self._addr   = address
        self._char   = char
        if calib is None:
            calib = IRCalib.load()
            if calib is None:
                print("IR calib: none saved — self-calibrating automatically as the car "
                      "drives (AutoIRCalib). Run ir_calib.py --calibrate instead for a "
                      "fixed, repeatable calibration.")
                calib = AutoIRCalib()
        self._reader = SensReader(calib)
        self._client = None
        self._err    = None
        self._write_fut = None

        self._loop   = asyncio.new_event_loop()
        self._ready  = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        if not self._ready.wait(connect_timeout + 5.0):
            raise RuntimeError("BLE connection timed out")
        if self._err is not None:
            raise self._err

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect())
        if self._err is None:
            self._loop.run_forever()

    async def _connect(self):
        try:
            self._client = BleakClient(self._addr)
            await self._client.connect()
            await self._client.start_notify(self._char, self._on_notify)
        except Exception as e:
            self._err = e
        finally:
            self._ready.set()

    def _on_notify(self, _char, data: bytearray):
        self._reader.feed(data)

    def send_cmd(self, vl: float, vr: float):
        # FFE1 only supports write-with-response (response=False is silently dropped).
        # Waiting for the ack would stall the 20 Hz loop, so skip this step if a
        # write is still in flight; the next step supersedes it.
        if self._client is None:
            return
        if self._write_fut is not None and not self._write_fut.done():
            return

        pkt = encode_cmd(vl, vr)
        self._write_fut = asyncio.run_coroutine_threadsafe(
            self._client.write_gatt_char(self._char, pkt, response=True), self._loop)

    def latest_sensors(self):
        return self._reader.get()

    def stop(self):
        # Drain the in-flight write first, or send_cmd() skips the stop command.
        if self._write_fut is not None:
            try:
                self._write_fut.result(timeout=1.0)
            except Exception:
                pass
            self._write_fut = None

        try:
            self.send_cmd(0.0, 0.0)
        except Exception:
            pass

        # Let the stop write finish before the loop is torn down.
        if self._write_fut is not None:
            try:
                self._write_fut.result(timeout=1.0)
            except Exception:
                pass

        async def _disconnect():
            try:
                if self._client is not None:
                    await self._client.stop_notify(self._char)
                    await self._client.disconnect()
            except Exception:
                pass

        try:
            asyncio.run_coroutine_threadsafe(_disconnect(), self._loop).result(timeout=5.0)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)


def make_ble_env(address: str | None, max_steps: int = 2000):
    """SerialHardwareEnv over a BleLink; scans for DEFAULT_NAME if address is None."""
    from hardware.pc_bridge_serial import SerialHardwareEnv

    if address is None:
        print(f"Scanning for a BLE device named {DEFAULT_NAME!r} …")
        address = discover()
        if address is None:
            raise RuntimeError(
                f"No BLE device named {DEFAULT_NAME!r} found. "
                f"Pass --address explicitly (see: python -m hardware.ble_scan).")
        print(f"Found {DEFAULT_NAME} at {address}")

    print(f"Connecting over BLE to {address} …")
    link = BleLink(address)
    return SerialHardwareEnv(max_steps=max_steps, link=link)
