import time

class GoodputMetrics:
    def __init__(self, window_size: int, mini_batch_size: int) -> None:
        self.window_size = window_size
        self.mini_batch_size = mini_batch_size
        self.last_loss = 0
        self.last_time = time.time()
        self.eps = 1e-6

    def reset_time(self) -> None:
        self.last_time = time.time()

    def _throughput(self, time) -> float:
        return (self.mini_batch_size * self.window_size) / (abs(time - self.last_time) + self.eps)

    def _statistical_efficiency(self, new_loss) -> float:
        return abs(new_loss - self.last_loss) / ((self.window_size * self.mini_batch_size) + self.eps)

    def _goodput(self, time, new_loss) -> float:
        return self._throughput(time) * self._statistical_efficiency(new_loss)

    def metrics(self, time, new_loss) -> dict:
        metrics = {
            "throughput": self._throughput(time),
            "statistical_efficiency": self._statistical_efficiency(new_loss),
            "goodput": self._goodput(time, new_loss),
        }
        self.last_loss = new_loss
        self.last_time = time
        return metrics
