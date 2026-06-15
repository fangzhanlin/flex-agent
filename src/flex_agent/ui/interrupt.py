from __future__ import annotations

import asyncio
import sys
import threading

if sys.platform != "win32":
    import select
    import termios
    import tty
else:
    select = None
    termios = None
    tty = None

class EscInterruptWatcher:
    """Watch stdin for Esc while an async agent turn is running."""

    def __init__(self) -> None:
        self._interrupted = asyncio.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def enabled(self) -> bool:
        return sys.stdin.isatty() and sys.platform != "win32"

    def start(self) -> None:
        if not self.enabled:
            return
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._watch, name="esc-interrupt", daemon=True)
        self._thread.start()

    def _watch(self) -> None:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        loop = self._loop
        if loop is None:
            return
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    loop.call_soon_threadsafe(self._interrupted.set)
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    async def wait(self) -> None:
        await self._interrupted.wait()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
