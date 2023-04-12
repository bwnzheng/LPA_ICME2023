import os
import neptune.new as neptune
from loguru import logger as loguru_logger

from logger.utils import NOP
from .configuration import load_yaml

logger = NOP()

class Logger:
    def __init__(self, rc=None, **kwargs) -> None:
        self.rc = dict() if rc is None else rc # runtime configurations
        self._init_rc()

        self.neprun = NOP()
        self.loguru = NOP()
        
        self.save_dict = dict()
    
    def init_neptune(self, enable, **kwargs):
        if not enable:
            return

        if self.rc.get('rank') is None or self.rc.get('rank') == 0:
            neptune_configs = load_yaml(self.rc['project_root'] / kwargs['config_file'])
            self.neprun = neptune.init_run(
                project=neptune_configs.get('project'),
                api_token=neptune_configs.get('api_token'),
                name=self.rc.get('exp_name'))

    def init_loguru(self, enable, **kwargs):
        if not enable:
            return

        self.loguru = loguru_logger
        self._config_loguru(**kwargs)

    def update_state(self, **kwargs):
        self.rc.update(kwargs)
    
    def update_save(self, **kwargs):
        for key in kwargs:
            if key not in self.save_dict:
                self.save_dict[key] = []

            self.save_dict[key].append(kwargs[key])

    def _log_nep(self, d: dict, base=()):
        for key in d:
            if isinstance(d[key], dict):
                self._log_nep(d[key], key)
            else:
                name = '/'.join(base + (key,))
                self.neprun[name].log(d[key])
    
    def _init_rc(self):
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            rank = 0
            world_size = 0
        self.rc.update(rank=rank, world_size=world_size)
        

    # customizations
    def _get_node_info(self):
        return f"rank[{self.rc.get('rank')}/{self.rc.get('world_size')-1}] "

    def _get_run_state(self):
        return f"{self.rc.get('state'):<5} ep[{self.rc.get('epoch_num')}/{self.rc.get('num_epochs')-1}] "

    def train_batch_start(self, **kwargs):
        batch = kwargs.get('batch')
        self.neprun['train/start_batch/batch_size'].log(len(batch))

    def train_batch_end(self, **kwargs):
        loss = kwargs['loss']
        self.neprun['train/batch/loss'].log(loss)

    def val_start(self, **kwargs):
        state_str = self._get_node_info() + self._get_run_state()
        self.loguru.info(state_str + f"validation start.")

    def val_end(self, **kwargs):
        d = {
            'val/acc': kwargs.get('acc'),
            'train/epoch/loss': kwargs.get('loss')
        }
        self._log_nep(d)

    def program_start(self, **kwargs):
        params = kwargs.get('params')
        self.loguru.info(self._get_node_info() + f'program start.')
        self.loguru.info(self._get_node_info() + str(params))

        self.neprun["parameters"] = params

    def program_end(self, **kwargs):
        self.loguru.info(self._get_node_info() + 'program end.')

        self.neprun.stop()

        self._save_to_disk()
    
    def _save_to_disk(self):
        pass

    def _config_loguru(self, **kwargs):
        pass


def init_logger(**kwargs):
    global logger
    logger = Logger(**kwargs)
    return logger
