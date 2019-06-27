from contextlib import ExitStack
from collections import deque


class ExitStackWithPop(ExitStack):
    def pop(self, cm):
        callbacks = self._exit_callbacks
        self._exit_callbacks = deque()
        found = None

        def unpack_cb(cb):
            if isinstance(cb, tuple):
                return cb[1]
            else:
                return cb

        while callbacks:
            cb = callbacks.popleft()
            if unpack_cb(cb).__self__ == cm:
                found = cb
            else:
                self._exit_callbacks.append(cb)
        if not found:
            raise KeyError("context manager not found")
        found = unpack_cb(found)
        found(None, None, None)
