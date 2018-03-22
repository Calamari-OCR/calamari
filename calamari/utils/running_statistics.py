class RunningStatistics:
    def __init__(self, max_size, initial_values=[]):
        self.max_size = max_size
        self.values = initial_values
        self.cur_sum = 0
        if len(initial_values) > 0:
            self.cur_sum = sum(self.values)

        if max_size < 1:
            raise Exception("A size > 0 is required. Got {}".format(max_size))

    def sum(self):
        return self.cur_sum

    def mean(self):
        return self.cur_sum / len(self.values)

    def push(self, v):
        self.values.append(v)

        if len(self.values) > self.max_size:
            self.cur_sum -= self.values[0]
            del self.values[0]

        self.cur_sum += v
