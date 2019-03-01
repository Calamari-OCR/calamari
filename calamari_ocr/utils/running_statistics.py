class RunningStatistics:
    def __init__(self, max_size, initial_values=None):
        self.max_size = max_size
        self.values = list(map(float, initial_values)) if initial_values is not None else []
        self.cur_sum = sum(self.values)

        if max_size < 1:
            raise Exception("A size > 0 is required. Got {}".format(max_size))

    def sum(self):
        return self.cur_sum

    def mean(self):
        return self.cur_sum / len(self.values)

    def push(self, v):
        v = float(v)
        self.values.append(v)

        if len(self.values) > self.max_size:
            self.cur_sum -= self.values[0]
            del self.values[0]

        self.cur_sum += v
