"""Streaming classification scores, weighted by sample count."""

import math


class ClassificationMetrics:
    def __init__(self, num_classes):
        self.matrix = [[0] * num_classes for _ in range(num_classes)]
        self.samples = 0
        self.loss_sum = 0.0

    def update(self, targets, predictions, loss_sum):
        if len(targets) != len(predictions):
            raise ValueError("Targets and predictions must have matching lengths")
        if not math.isfinite(loss_sum) or loss_sum < 0:
            raise ValueError("Cross entropy must be finite and nonnegative")
        for target, prediction in zip(targets, predictions):
            if not (0 <= target < len(self.matrix) and 0 <= prediction < len(self.matrix)):
                raise ValueError("Class index out of range")
        for target, prediction in zip(targets, predictions):
            self.matrix[target][prediction] += 1
        self.samples += len(targets)
        self.loss_sum += loss_sum

    def compute(self):
        if not self.samples:
            raise ValueError("Cannot score an empty dataset")
        return {
            "samples": self.samples,
            "accuracy": sum(row[i] for i, row in enumerate(self.matrix)) / self.samples,
            "cross_entropy_loss": self.loss_sum / self.samples,
            "confusion_matrix": self.matrix,
        }

