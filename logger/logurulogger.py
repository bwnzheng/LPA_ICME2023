import sys

from loguru import logger as loguru_logger

from .rtlogger import RTLogger
from .utils import NOP

class LoguruLogger(RTLogger):
    def __init__(self) -> None:
        super().__init__()

        self.loguru = NOP()
        
        self._init_loguru()
    
    def node_info(self):
        return self._get_node_info()
    
    def task_info(self):
        return self._get_task_info()
    
    def epoch_info(self):
        return self._get_run_epoch()

    def batch_info(self):
        return self._get_run_batch()
        
    # customizations
    def _init_loguru(self):
        print('init loguru')
        self.loguru = loguru_logger

        if not self.rc.get('enable_logfile'):
            return

        log_dir = self.rc.get('log_dir')
        if log_dir is not None:
            rank = self.rc.get('rank', 0)

            filename = f'stdlog{rank}.log'
            self.loguru.add(log_dir / filename, level='INFO')

            if self.rc.get('debug'):
                debug_filename = f'debuglog{rank}.log'
                self.loguru.add(log_dir / debug_filename, level=0)

            if rank == 0:
                performance_logname = 'acclog.log'
                self.loguru.add(log_dir / performance_logname, level='SUCCESS')
