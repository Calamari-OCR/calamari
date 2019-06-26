from contextlib import ExitStack
from collections import deque


class ExitStackWithPop(ExitStack):
    def pop(self, cm):
        callbacks = self._exit_callbacks
        self._exit_callbacks = deque()
        found = None
        while callbacks:
            cb = callbacks.popleft()
            # Newer versions return a (bool, cb) tuple
            if isinstance(cb, tuple):
                # search for the cb (should be the last element)
                for c in cb:
                    try:
                        if c.__self__ == cm:
                            found = c
                            break
                    except AttributeError:
                        continue
            # Older version return the cb directly
            elif cb.__self__ == cm:
                found = cb
            else:
                self._exit_callbacks.append(cb)
        if not found:
            raise KeyError("context manager not found")
        found(None, None, None)