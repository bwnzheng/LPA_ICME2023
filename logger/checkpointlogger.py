import pickle
from .rtlogger import RTLogger, update_state


class CheckpointLogger(RTLogger):
    def __init__(self, filename='ckpt') -> None:
        super().__init__()

        self.ckpt_dir = self.rc['ckpt_dir']
        self.filename = filename

    def _get_ckpt_info(self):
        task_info = self._get_task_info()
        return 'task' + task_info.replace('/', '-')

    def save_to_disk(self, **kwargs):
        # save all the kwargs and current state to one file in ckpt_dir.
        save_dict = dict(
            state = self.state,
            data = kwargs
        )
        ckpt_info = self._get_ckpt_info()

        filename = self.filename + ckpt_info + '.ckpt'
        filepath = self.ckpt_dir / filename

        with filepath.open('wb') as f:
            pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)

    def load_from_disk(self, **kwargs):
        ckpt_info = self._get_ckpt_info()
        filename = self.filename + ckpt_info + '.ckpt'
        filepath = self.ckpt_dir / filename

        try:
            with filepath.open('rb') as f:
                load_dict = pickle.load(f)
        except FileNotFoundError:
            return False
        
        data = load_dict['data']
        update_state(**load_dict['state'])
        update_state(run_state='prepare')
        
        for k, v in data.items():
            if k in kwargs:
                kwargs[k].load_state_dict(v)
        
        return True
