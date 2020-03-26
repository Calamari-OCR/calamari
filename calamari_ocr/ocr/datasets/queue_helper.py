from multiprocessing.queues import JoinableQueue
from queue import Full


class FeedBackJoinableQueue(JoinableQueue):
    def __init__(self, maxsize, max_feed, ctx):
        super().__init__(maxsize=maxsize, ctx=ctx)
        self.max_feed_queue = max_feed

    def get(self, block=True, timeout=None):
        r = super().get(block, timeout)
        self.max_feed_queue.job_finished()
        return r


class MaxFeedJoinablQueue(JoinableQueue):
    def __init__(self, maxsize, ctx):
        super().__init__(maxsize=maxsize, ctx=ctx)
        self._cursize = ctx.Value('i', 0)

    def __getstate__(self):
        return JoinableQueue.__getstate__(self) + (self._cursize, )

    def __setstate__(self, state):
        JoinableQueue.__setstate__(self, state[:-1])
        self._cursize, = state[-1:]

    def job_finished(self):
        with self._cond:
            if self._cursize.value > 0:
                self._cursize.value -= 1

    def put(self, obj, block=True, timeout=None):
        with self._cond:
            if self._cursize.value >= self._maxsize:
                raise Full
            self._cursize.value += 1

        return super().put(obj, block, timeout)


class MaxElementsQueuer:
    def __init__(self, maxsize, ctx):
        # the input queue tracks how many elements are queued and only enqueues new elements if the number of elements
        # in all queues is smaller than maxsize
        self.input_queue = MaxFeedJoinablQueue(maxsize, ctx)
        self.output_queue = FeedBackJoinableQueue(maxsize, self.input_queue, ctx)

