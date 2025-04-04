import time

class GoodputMetrics:
    def __init__(self, window_size: int, grad_acc_steps: int, mini_batch_size: int) -> None:
        self.grad_acc_steps = grad_acc_steps
        self.mini_batch_size = mini_batch_size
        self.global_batch_size = self.mini_batch_size * self.grad_acc_steps # Samples per step

        self.last_loss = 0.0
        self.start_time = time.time()
        self.last_end_time = self.start_time
        self.eps = 1e-6
        self.initialized = False
        self.window_size = window_size

    def reset_time(self) -> None:
        self.start_time = time.time()
        self.last_end_time = self.start_time
        self.initialized = False

    # Standard Throughput: Samples / sec
    def _throughput(self, current_end_time) -> float:
        duration = (current_end_time - self.last_end_time) + self.eps
        return (self.global_batch_size * self.window_size) / duration

    # Standard Statistical Efficiency: Loss change / sample
    def _statistical_efficiency(self, new_loss) -> float:
        loss_change = abs(new_loss - self.last_loss)
        return loss_change / (self.window_size * (self.global_batch_size + self.eps))

    # Goodput: Throughput * SE = (Samples/sec) * (Loss change / Sample) = Loss change / sec
    def _goodput(self, current_end_time, new_loss) -> float:
        # Method 1: Definition
        return self._throughput(current_end_time) * self._statistical_efficiency(new_loss)

    def metrics(self, current_end_time, new_loss) -> dict:
        # Calculate metrics *before* updating state
        throughput = self._throughput(current_end_time) # Standard definition
        statistical_efficiency = self._statistical_efficiency(new_loss) # Standard definition
        goodput = self._goodput(current_end_time, new_loss) # loss_change / sec

        # Update state
        self.last_loss = new_loss
        self.last_end_time = current_end_time
        self.initialized = True

        metrics = {
            "throughput": throughput,
            "statistical_efficiency": statistical_efficiency,
            "goodput": goodput,
        }
        return metrics