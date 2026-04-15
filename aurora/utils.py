import os
import logging
import sys
import functools
from typing import Any,Iterable
import pynvml
import torch
import numpy as np

def get_rs(x = None) -> np.random.RandomState:
    if isinstance(x, int):
        return np.random.RandomState(x)
    if isinstance(x, np.random.RandomState):
        return x
    return np.random

def get_default_numpy_dtype() -> type:
    return getattr(np, str(torch.get_default_dtype()).replace("torch.", ""))


class _CriticalFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING

class _NonCriticalFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING
    
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

def get_chained_attr(x: Any, attr: str) -> Any:
    for k in attr.split("."):
        if not hasattr(x, k):
            raise AttributeError(f"{attr} not found!")
        x = getattr(x, k)
    return x

class LogManager(metaclass=SingletonMeta):

    r"""
    Manage loggers used in the package
    """

    def __init__(self) -> None:
        self._loggers = {}
        self._log_file = None
        self._console_log_level = logging.INFO
        self._file_log_level = logging.DEBUG
        self._file_fmt = \
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
        self._console_fmt = \
            "[%(levelname)s] %(name)s: %(message)s"
        self._date_fmt = "%Y-%m-%d %H:%M:%S"

    @property
    def log_file(self) -> str:
        r"""
        Configure log file
        """
        return self._log_file

    @property
    def file_log_level(self) -> int:
        r"""
        Configure logging level in the log file
        """
        return self._file_log_level

    @property
    def console_log_level(self) -> int:
        r"""
        Configure logging level printed in the console
        """
        return self._console_log_level

    def _create_file_handler(self) -> logging.FileHandler:
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.file_log_level)
        file_handler.setFormatter(logging.Formatter(
            fmt=self._file_fmt, datefmt=self._date_fmt))
        return file_handler

    def _create_console_handler(self, critical: bool) -> logging.StreamHandler:
        if critical:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.addFilter(_CriticalFilter())
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.addFilter(_NonCriticalFilter())
        console_handler.setLevel(self.console_log_level)
        console_handler.setFormatter(logging.Formatter(fmt=self._console_fmt))
        return console_handler

    def get_logger(self, name: str) -> logging.Logger:
        r"""
        Get a logger by name
        """
        if name in self._loggers:
            return self._loggers[name]
        new_logger = logging.getLogger(name)
        new_logger.setLevel(logging.DEBUG)  # lowest level
        new_logger.addHandler(self._create_console_handler(True))
        new_logger.addHandler(self._create_console_handler(False))
        if self.log_file:
            new_logger.addHandler(self._create_file_handler())
        self._loggers[name] = new_logger
        return new_logger

    @log_file.setter
    def log_file(self, file_name: os.PathLike) -> None:
        self._log_file = file_name
        for logger in self._loggers.values():
            for idx, handler in enumerate(logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    logger.handlers[idx].close()
                    if self.log_file:
                        logger.handlers[idx] = self._create_file_handler()
                    else:
                        del logger.handlers[idx]
                    break
            else:
                if file_name:
                    logger.addHandler(self._create_file_handler())

    @file_log_level.setter
    def file_log_level(self, log_level: int) -> None:
        self._file_log_level = log_level
        for logger in self._loggers.values():
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.setLevel(self.file_log_level)
                    break

    @console_log_level.setter
    def console_log_level(self, log_level: int) -> None:
        self._console_log_level = log_level
        for logger in self._loggers.values():
            for handler in logger.handlers:
                if type(handler) is logging.StreamHandler:  # pylint: disable=unidiomatic-typecheck
                    handler.setLevel(self.console_log_level)

log = LogManager()

def logged(obj):
    r"""
    Add logger as an attribute
    """
    obj.logger = log.get_logger(obj.__name__)
    return obj

@logged
class ConfigManager:
    def __init__(self) -> None:
        self.TMP_PREFIX = "AuroraTMP"
        self.ANNDATA_KEY = "__AuroraConfig__"
        self.CPU_ONLY = False
        self.CUDNN_MODE = "repeatability"
        self.MASKED_GPUS = []
        self.DATALOADER_PIN_MEMORY = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.DATALOADER_FETCHES_PER_BATCH = 4

config = ConfigManager()

@logged
@functools.lru_cache(maxsize=1)
def autodevice() -> torch.device:
    used_device = -1
    if not config.CPU_ONLY:
        try:
            pynvml.nvmlInit()
            free_mems = np.array([
                pynvml.nvmlDeviceGetMemoryInfo(
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                ).free for i in range(pynvml.nvmlDeviceGetCount())
            ])
            for item in config.MASKED_GPUS:
                free_mems[item] = -1
            best_devices = np.where(free_mems == free_mems.max())[0]
            used_device = np.random.choice(best_devices, 1)[0]
            if free_mems[used_device] < 0:
                used_device = -1
        except pynvml.NVMLError:
            pass
    if used_device == -1:
        autodevice.logger.info("Using CPU as computation device.")
        return torch.device("cpu")
    autodevice.logger.info("Using GPU %d as computation device.", used_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(used_device)
    return torch.device("cuda")

@logged
def prod(x: Iterable):
    try:
        from math import prod
        return prod(x)
    except ImportError:
        ans = 1
        for item in x:
            ans = ans * item
        return ans
