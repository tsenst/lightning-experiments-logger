import time
from multiprocessing import Pipe, Process, Queue

import pynvml
from pytorch_lightning.callbacks import Callback
from typing import List
import numpy as np

from enum import Enum

class Stage(Enum):
    TRAIN = "train"
    TRAIN_EPOCH = "train_epoch"
    TRAIN_BATCH = "train_batch"
    VALIDATION = "validation"
    VALIDATION_EPOCH = "validation_epoch"
    VALIDATION_BATCH = "validation_batch"
    TEST = "test"
    TEST_EPOCH = "test_epoch"
    TEST_BATCH = "test_batch"
    PREDICT = "predict"
    PREDICT_EPOCH = "predict_epoch"
    PREDICT_BATCH = "predict_batch"

class DeviceLogger:
    def __init__(self, handle, sample_rate: float):
        self._process = None
        self._handle = handle
        self._process: Process
        self._sample_rate = sample_rate
        self._metrics = {"memory": [], "gpu": [], "energy": [], "markers": {}}

    def start_logging(self) -> None:
        self._parent_conn, self._conn = Pipe()
        self._message_queue = Queue()
        self._data_queue = Queue()
        self._data_queue.put(self._metrics)
        self._process = Process(
            target=self._log_metric,
            args=(self._message_queue, self._data_queue),
        )
        self._process.start()

    def stop_logging(self) -> None:
        self._message_queue.put("quit")
        self._process.join()
        self._metrics = self._data_queue.get()

    def set_marker(self, marker: str) -> None:
        self._message_queue.put(marker)

    def _log_metric(self, message_queue: Queue, data_queue: Queue) -> None:
        marker_value = ""
        metrics = data_queue.get()
        while marker_value != "quit":

            time.sleep(self._sample_rate)

            rates = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            energy_consuption = pynvml.nvmlDeviceGetTotalEnergyConsumption(
                self._handle
            )

            metrics["memory"].append(rates.memory)
            metrics["gpu"].append(rates.gpu)
            metrics["energy"].append(energy_consuption)
            if not message_queue.empty():
                marker_value = message_queue.get()
                idx = len(metrics["memory"]) - 1
                if marker_value != "quit":
                    if marker_value not in metrics["markers"]:
                        metrics["markers"][idx] = [marker_value]
                    else:
                        metrics["markers"][idx].append(marker_value)

        data_queue.put(metrics)

    def filter_metrics(self, start_marker: str, end_marker: str) -> List[List]:
        metric_names = ["gpu", "energy", "memory"]
        for metric_name in metric_names:
            self._metrics[metric_name] = np.asarray(self._metrics[metric_name])

        out_metrics = {metric_name: [] for metric_name in metric_names}
        from_idx = None

        for idx, markers in self._metrics["markers"].items():
            if start_marker in markers:
                from_idx = idx
            if end_marker in markers:
                for metric_name in metric_names:
                    out_metrics[metric_name].append(
                        self._metrics[metric_name][from_idx:idx]
                    )

        return out_metrics

    def get_metrics(self, stage: Stage) -> None:
        return self._filter_metrics(
            start_marker="on_" + stage.value + "_start",
            end_marker="on_" + stage.value + "_end",
        )


class GPUMonitorCallback(Callback):
    def __init__(self, sample_rate: float):
        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        self.devices_logger = [
            DeviceLogger(
                handle=pynvml.nvmlDeviceGetHandleByIndex(i),
                sample_rate=sample_rate,
            )
            for i in range(deviceCount)
        ]
        pynvml.nvmlShutdown()

    def setup(self, **kwargs) -> None:
        pynvml.nvmlInit()
        for devices in self.devices_logger:
            devices.start_logging()

    def teardown(self, **kwargs) -> None:
        for devices in self.devices_logger:
            devices.stop_logging()
        pynvml.nvmlShutdown()

    def on_train_start(self, **kwargs) -> None:
        self.set_marker(marker="on_train_start")

    def on_train_end(self, **kwargs) -> None:
        self.set_marker(marker="on_train_end")

    def on_validation_start(self, **kwargs) -> None:
        self.set_marker(marker="on_validation_start")

    def on_validation_end(self, **kwargs) -> None:
        self.set_marker(marker="on_validation_end")

    def on_test_start(self, **kwargs) -> None:
        self.set_marker(marker="on_test_start")

    def on_test_end(self, **kwargs) -> None:
        self.set_marker(marker="on_test_end")

    def on_predict_start(self, **kwargs) -> None:
        self.set_marker(marker="on_predict_start")

    def on_predict_end(self, **kwargs) -> None:
        self.set_marker(marker="on_predict_end")

    def on_train_epoch_start(self, **kwargs) -> None:
        self.set_marker(marker="on_train_epoch_start")

    def on_train_epoch_end(self, **kwargs) -> None:
        self.set_marker(marker="on_train_epoch_end")

    def on_validation_epoch_start(self, **kwargs) -> None:
        self.set_marker(marker="on_validation_epoch_start")

    def on_validation_epoch_end(self, **kwargs) -> None:
        self.set_marker(marker="on_validation_epoch_end")

    def on_test_epoch_start(self, **kwargs) -> None:
        self.set_marker(marker="on_test_epoch_start")

    def on_test_epoch_end(self, **kwargs) -> None:
        self.set_marker(marker="on_test_epoch_end")

    def on_predict_epoch_start(self, **kwargs) -> None:
        self.set_marker(marker="on_predict_epoch_start")

    def on_predict_epoch_end(self, **kwargs) -> None:
        self.set_marker(marker="on_predict_epoch_end")

    def on_train_batch_start(self, **kwargs) -> None:
        self.set_marker(marker="on_train_batch_start")

    def on_train_batch_end(self, **kwargs) -> None:
        self.set_marker(marker="on_train_batch_end")

    def on_validation_batch_start(self, **kwargs) -> None:
        self.set_marker(marker="on_validation_batch_start")

    def on_validation_batch_end(self, **kwargs) -> None:
        self.set_marker(marker="on_validation_batch_end")

    def on_test_batch_start(self, **kwargs) -> None:
        self.set_marker(marker="on_test_batch_start")

    def on_test_batch_end(self, **kwargs) -> None:
        self.set_marker(marker="on_test_batch_end")

    def on_before_backward(self, **kwargs) -> None:
        self.set_marker(marker="on_before_backward")

    def on_after_backward(self, **kwargs) -> None:
        self.set_marker(marker="on_after_backward")

    def on_before_optimizer_step(self, **kwargs) -> None:
        self.set_marker(marker="on_before_optimizer_step")

    def on_before_zero_grad(self, **kwargs) -> None:
        self.set_marker(marker="on_before_zero_grad")

    def set_marker(self, marker: str) -> None:
        for devices in self.devices_logger:
            devices.set_marker(marker=marker)

