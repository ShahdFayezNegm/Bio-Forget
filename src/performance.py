import time

class PerformanceAnalyzer:
    def __init__(self, full_retraining_time=60):
        self.full_retraining_time = full_retraining_time

    def compare_unlearning_vs_retraining(self, unlearning_time):
        speedup = (
            self.full_retraining_time / unlearning_time
            if unlearning_time > 0 else 0
        )

        return {
            "full_retraining_time": self.full_retraining_time,
            "unlearning_time": unlearning_time,
            "speedup_factor": speedup
        }