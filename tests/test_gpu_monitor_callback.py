from typing import Any
from experiments_addon.gpu_monitor_callback import GPUMonitorCallback
from unittest.mock import patch
import time

UTIL_COUNTER = 0
ENERGY_COUNTER = 0
class MockRates:
    def __init__(self, gpu: float, memory: float):
        self.gpu = gpu
        self.memory = memory


def mock_nvmlDeviceGetUtilizationRates(handle: Any):
    global UTIL_COUNTER
    UTIL_COUNTER += 1
    return MockRates(gpu=UTIL_COUNTER * 0.1, memory=UTIL_COUNTER*10)

def mock_nvmlDeviceGetTotalEnergyConsumption(handle: Any):
    global ENERGY_COUNTER
    ENERGY_COUNTER +=1
    return ENERGY_COUNTER



def test_GPUMonitorCallback() -> None:
    with patch("pynvml.nvmlInit"), \
        patch("pynvml.nvmlShutdown"), \
        patch("pynvml.nvmlDeviceGetCount", return_value=1), \
        patch("pynvml.nvmlDeviceGetHandleByIndex", return_value="index"), \
        patch("pynvml.nvmlDeviceGetUtilizationRates", wraps=mock_nvmlDeviceGetUtilizationRates), \
        patch("pynvml.nvmlDeviceGetTotalEnergyConsumption", wraps=mock_nvmlDeviceGetTotalEnergyConsumption):
        callback = GPUMonitorCallback(sample_rate=0.1)
        callback.setup()
        time.sleep(0.1)
        callback.on_train_start()
        callback.on_train_batch_start()
        time.sleep(0.2)
        callback.teardown()
        callback.setup()
        time.sleep(0.2)
        callback.on_test_start()
        callback.teardown()
    assert callback.devices_logger[0]._metrics == {
        "memory": [10, 20, 30, 10, 20, 30],
        "gpu": [0.1, 0.2, 0.30000000000000004, 0.1, 0.2, 0.30000000000000004],
        "energy": [1, 2, 3, 1, 2, 3],
        "markers": {
            0: ["on_train_start"],
            1: ["on_train_batch_start"],
            4: ["on_test_start"]
        }
    }
