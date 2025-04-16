import time
import statistics as st

class GoodputMetrics:
    def __init__(self, window_size: int, grad_acc_steps: int, mini_batch_size: int) -> None:
        self.grad_acc_steps = grad_acc_steps
        self.mini_batch_size = mini_batch_size
        self.global_batch_size = self.mini_batch_size * self.grad_acc_steps # Samples per step

        self.losses = []
        self.last_loss = 0.0
        self.start_time = time.time()
        self.last_end_time = self.start_time
        self.eps = 1e-6
        self.initialized = False
        self.window_size = window_size
        self.avg_goodput= []
        self.avg_stat_efficiency = []
        self.avg_throughput = []

    def reset_time(self) -> None:
        self.start_time = time.time()
        self.last_end_time = self.start_time
        self.initialized = False

    def _throughput(self, current_end_time) -> float:
        duration = (current_end_time - self.last_end_time) + self.eps
        return (self.global_batch_size * self.window_size) / duration

    def _statistical_efficiency(self, new_loss) -> float:
        self.losses.append(new_loss)
        if len(self.losses) > self.window_size:
            loss_change = abs(self.losses[-1] - self.losses[-(self.window_size + 1)])
            return loss_change / ((self.window_size * self.global_batch_size) + self.eps)
        else:
            return 1.0

    def _goodput(self, current_end_time, new_loss) -> float:
        return self._throughput(current_end_time) * self._statistical_efficiency(new_loss)

    def metrics(self, current_end_time, new_loss) -> dict:
        throughput = self._throughput(current_end_time) 
        statistical_efficiency = self._statistical_efficiency(new_loss)
        goodput = throughput * statistical_efficiency

        self.avg_throughput.append(throughput)
        self.avg_stat_efficiency.append(statistical_efficiency)
        self.avg_goodput.append(goodput)

        self.last_loss = new_loss
        self.last_end_time = current_end_time
        self.initialized = True

        metrics = {
            "throughput": throughput,
            "statistical_efficiency": statistical_efficiency,
            "goodput": goodput,
            "avg_throughput": st.mean(self.avg_throughput),
            "avg_statistical_efficiency": st.mean(self.avg_stat_efficiency),
            "avg_goodput": st.mean(self.avg_goodput),
        }
        return metrics