
from .logurulogger import LoguruLogger
from .neptunelogger import NeptuneLogger
from .savelogger import SaveLogger
from .checkpointlogger import CheckpointLogger

from .utils import NOP

class Loggers:
    def __init__(self) -> None:
        
        self._neptune_logger = NOP()
        self._save_logger = NOP()
        self._loguru_logger = NOP()
        self._checkpoint_logger = NOP()

    @property
    def neptune(self):
        return self._neptune_logger
    
    def init_neptune(self):
        self._neptune_logger = NeptuneLogger()
    
    @property
    def save(self):
        return self._save_logger

    def init_savelogger(self):
        self._save_logger = SaveLogger()
    
    @property
    def loguru_logger(self):
        return self._loguru_logger
    
    @property
    def loguru(self):
        return self._loguru_logger.loguru

    def init_loguru(self):
        self._loguru_logger = LoguruLogger()
    
    @property
    def checkpoint(self):
        return self._checkpoint_logger
    
    def init_checkpoint(self):
        self._checkpoint_logger = CheckpointLogger(filename='ckpt_')

    