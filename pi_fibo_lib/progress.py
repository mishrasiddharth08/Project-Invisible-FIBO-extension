"""One-second native status heartbeat. Never invent steps or decode extra previews."""
from threading import Event, Lock, Thread
from time import monotonic

class LiveProgress:
    def __init__(self, state, interval=1.0):
        self.state = state
        self.interval = interval
        self._done = Event()
        self._lock = Lock()
        self._phase = "Loading model"
        self._since = monotonic()
        self._thread = None

    def phase(self, label):
        with self._lock:
            if label != self._phase:
                self._phase, self._since = label, monotonic()
        self.tick()

    def tick(self):
        with self._lock:
            label = self._phase
            elapsed = int(monotonic() - self._since)
        if self.state.interrupted or getattr(self.state, "stopping_generation", False):
            label = "Stopping at the next safe point"
        self.state.textinfo = f"FIBO: {label} · {elapsed}s · step {self.state.sampling_step}/{self.state.sampling_steps}"

    def _run(self):
        while not self._done.wait(self.interval):
            self.tick()

    def __enter__(self):
        self.tick()
        self._thread = Thread(target=self._run, name="FIBO-progress", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._done.set()
        self._thread.join()
        self.state.textinfo = "FIBO stopped" if self.state.interrupted else ("FIBO failed" if exc[0] else "FIBO complete")
