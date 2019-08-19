from typing import Tuple

from clfw import TaskSequence
import numpy as np
from tensorflow.python.data import Dataset
from tensorflow_datasets import as_numpy


def dataset_to_numpy(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    x_total = y_total = None
    for x, y in as_numpy(dataset):
        if x_total is None:
            x_total, y_total = x, y
        else:
            x_total = np.vstack((x_total, x))
            y_total = np.vstack((y_total, y))
    return x_total, y_total


class Generator:
    def __init__(self, task_sequence: TaskSequence):
        self.task_sequence = task_sequence
        self.index = 0
        self.max_iter = task_sequence.ntasks

    def get_dims(self):
        ts = self.task_sequence
        return ts.feature_dim, ts.nlabels

    def next_task(self):
        training_set = self.task_sequence.training_sets[self.index]
        x_train, y_train = dataset_to_numpy(training_set)
        valid_set = self.task_sequence.validation_sets[self.index]
        x_valid, y_valid = dataset_to_numpy(valid_set)
        test_set = self.task_sequence.test_sets[self.index]
        x_test, y_test = dataset_to_numpy(test_set)
        self.index += 1
        return x_train, y_train, x_valid, y_valid, x_test, y_test
