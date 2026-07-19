from app.config import Device
from app.domain.errors import ConfigurationError


def select_device(requested: Device) -> str:
    import torch

    if requested is Device.AUTO:
        if torch.cuda.is_available():
            return Device.CUDA.value
        if torch.backends.mps.is_available():
            return Device.MPS.value
        return Device.CPU.value
    if requested is Device.CUDA and not torch.cuda.is_available():
        raise ConfigurationError("CUDA was requested but is unavailable")
    if requested is Device.MPS and not torch.backends.mps.is_available():
        raise ConfigurationError("MPS was requested but is unavailable")
    return requested.value
