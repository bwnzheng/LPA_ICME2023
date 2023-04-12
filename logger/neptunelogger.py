import os
from .rtlogger import RTLogger
from .utils import NOP
from modules.configuration import load_yaml

class NeptuneLogger(RTLogger):
    def __init__(self) -> None:
        super().__init__()

        self.neprun = NOP()
        self._init_neptune()
    
    def _init_neptune(self):
        if not self.rc.get('enable_neptune'): # False or no such field
            return

        if self.rc.get('rank') is None or self.rc.get('rank') == 0:
            import neptune
            neptune_configs = load_yaml(self.rc['neptune_config'])
            self.neprun = neptune.new.init_run(
                project=neptune_configs.get('project'),
                api_token=neptune_configs.get('api_token'),
                name=self.rc.get('exp_name'),
                custom_run_id=self.rc.get('neptune_exp_id'),
                proxies={
                    'http': os.environ['http_proxy'], 
                    'https': os.environ['https_proxy']})
            
            self.neprun['parameters'] = self.rc
    
    def log_run_state(self):
        task_info = f"{self.state.get('task_num')}/{self.state.get('num_tasks')}"
        epoch_info = f"{self.state.get('epoch_num')}/{self.state.get('num_epochs')-1}"
        run_state = self.state.get('run_state')

        self.neprun['state/task'] = task_info
        self.neprun['state/epoch'] = epoch_info
        self.neprun['state/run_state'] = run_state

    
    def log_nep(self, d: dict, base=()):
        for key in d:
            if isinstance(d[key], dict):
                self.log_nep(d[key], key)
            else:
                name = '/'.join(base + (key,))
                self.neprun[name].append(d[key])
