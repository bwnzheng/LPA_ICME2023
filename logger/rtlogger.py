from functools import partial
from abc import abstractmethod

_rc = dict()
_state = dict()

def update_state(**kwargs):
    _state.update(kwargs)

def add_state(**kwargs):
    for k, v in kwargs.items():
        if k not in _state:
            _state[k] = 0
        
        _state[k] += v

def update_rc(**kwargs):
    _rc.update(kwargs)

def init_rc(args):
    _rc.update(vars(args))

# Runtime Logger
class RTLogger:
    def __init__(self, rc=_rc, state=_state) -> None: 
        self._rc = rc # runtime configurations
        self._state = state # running state
    
    @property
    def state(self):
        return self._state

    @property
    def rc(self):
        return self._rc

    def _get_node_info(self):
        if self.rc.get('distributed'):
            return \
            f"rank[{self.rc.get('rank')}/"\
            f"{self.rc.get('world_size')-1}]"
        else:
            return ''
    
    def _get_run_state(self):
        return \
            f"{self.state.get('run_state'):^7}"
    
    def _get_task_info(self):
        return \
            f"[{self.state.get('task_num')}/"\
            f"{self.state.get('num_tasks')}]"
    
    def _get_epoch_info(self):
        return \
            f"[{self.state.get('epoch_num')}/"\
            f"{self.state.get('num_epochs')-1}]"\
    
    def _get_batch_info(self):
        return \
            f"[{self.state.get('batch_num')}/"\
            f"{self.state.get('num_batches')}]"

    def _get_run_task(self):
        node_info = self._get_node_info()
        state_info = self._get_run_state()
        task_info = self._get_task_info()
        return node_info + state_info + task_info

    def _get_run_epoch(self):
        node_info = self._get_node_info()
        state_info = self._get_run_state()
        task_info = self._get_task_info()
        epoch_info = self._get_epoch_info()
        return node_info + state_info + task_info + epoch_info

    def _get_run_batch(self):
        node_info = self._get_node_info()
        state_info = self._get_run_state()
        task_info = self._get_task_info()
        epoch_info = self._get_epoch_info()
        batch_info = self._get_batch_info()
        return node_info + state_info + task_info + epoch_info + batch_info
    
    def _get_task_step(self): # total number of tasks
        task_num = self.state['task_num']
        return task_num
    
    def _get_epoch_step(self): # total number of epochs
        total_epochs = self.state['total_epochs']
        return total_epochs - 1

    def _get_batch_step(self): # total number of batches
        total_batches = self.state['total_batches']
        return total_batches - 1


class StdLogger(RTLogger):
    def __init__(self) -> None:
        super().__init__()
    
    def __getattr__(self, __name: str):
        if __name.startswith('node_'):
            level = __name.removeprefix('node_')
            return partial(self._out_with_nodeinfo, level)
        elif __name.startswith('state_'):
            level = __name.removeprefix('state_')
            return partial(self._out_with_batchinfo, level)
        else:
            raise AttributeError(f"'{__class__.__name__}' object has no attribute '{__name}'")
    
    def _out_with_nodeinfo(self, level: str, text: str):
        out_str = ' '.join((self._get_node_info(), text))
        self.out(out_str, level)
    
    def _out_with_batchinfo(self, level: str, text: str):
        out_str = ' '.join((self._get_node_info(), self._get_run_batch(), text))
        self.out(out_str, level)
    
    @abstractmethod
    def out(self, text: str, level: str=''):
        raise NotImplementedError()


class StdPrintLogger(StdLogger):
    def __init__(self) -> None:
        super().__init__()

    def out(self, text: str, level: str = ''):
        out_str = ' '.join(level, text)
        print(out_str)
