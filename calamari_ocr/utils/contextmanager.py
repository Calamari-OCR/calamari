from contextlib import ExitStack
from collections import deque

class ExitStackWithPop(ExitStack):
    def pop(self, cm):
        callbacks = self._exit_callbacks
        self._exit_callbacks = deque()
        found = None
        while callbacks:
            cb = callbacks.popleft()
            if cb.__self__ == cm:
                found = cb
            else:
                self._exit_callbacks.append(cb)
        if not found:
            raise KeyError("context manager not found")
        found(None, None, None)