import time

# fully customizable
from . import loggers as lgrs

def prepare_backbone(**kwargs):
    backbone_model = kwargs['backbone_model']

    parameter_count = sum(p.numel() for p in backbone_model.parameters() if p.requires_grad)

    lgrs.loguru.info(f'number of parameters in backbone: {parameter_count}.')
    lgrs.loguru.info(f'backbone: {backbone_model}')

def train_epoch_start(**kwargs):
    if _is_main_process():
        lgrs.neptune.log_run_state()
        
    lgrs.loguru_logger.state['epoch_start_time'] = time.time()

def train_epoch_end(**kwargs):
    metric_logger = kwargs['metric_logger']

    epoch_info = lgrs.loguru_logger.epoch_info()
    epoch_time = time.time() - lgrs.loguru_logger.state['epoch_start_time']

    info_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    infos = [f'{k}: {v:>.4}' for k, v in info_dict.items()]
    infos.append(f'total time: {epoch_time:>.4}s')
    
    # infos = [f'{k}: {meter.global_avg:>.4}' for k, meter in metric_logger.meters.items()]
    msg1 = ' '.join((epoch_info, 'Epoch Gathered Averaged Stats:', *infos))
    lgrs.loguru.info(msg1)

    if _is_main_process():
        epoch_step = lgrs.loguru_logger._get_epoch_step()
        for k, v in info_dict.items():
            lgrs.neptune.neprun[f'train/epoch/{k}'].append(v, step=epoch_step)
        
        lgrs.neptune.neprun['time/epoch'].append(epoch_time, step=epoch_step)


def train_first_batch_start(**kwargs):
    batch_info = lgrs.loguru_logger.batch_info()
    samples = kwargs['samples']

    msg1 = ' '.join((batch_info, f'Batch size is {samples.shape}'))
    lgrs.loguru.trace(msg1)

def train_batch_start(**kwargs):
    # batch_info = lgrs.loguru_logger.batch_info()
    # samples = kwargs['samples']

    # msg1 = ' '.join((batch_info, f'Batch size is {samples.shape}'))
    # lgrs.loguru.info(msg1)
    pass

def train_batch_end_with_interval(interval=10, **kwargs):
    batch_num = lgrs.loguru_logger.state['batch_num']
    num_batches = lgrs.loguru_logger.state['num_batches']
    if batch_num % interval == 0 or batch_num == num_batches - 1:
        batch_info = lgrs.loguru_logger.batch_info()
        infos = [f'{k}: {v:.4}' for k, v in kwargs.items()]
        msg1 = ' '.join((batch_info, *infos))
        lgrs.loguru.info(msg1)

        if _is_main_process():
            batch_step = lgrs.loguru_logger._get_batch_step()
            for k, v in kwargs.items():
                lgrs.neptune.neprun[f'train/batch/{k}'].append(v, step=batch_step)

def eval_epoch_start(**kwargs):
    epoch_info = lgrs.loguru_logger.epoch_info()
    msg1 = ' '.join((epoch_info, 'epoch eval start.'))
    lgrs.loguru.info(msg1)

    if _is_main_process():
        lgrs.neptune.log_run_state()

def eval_epoch_end(**kwargs):
    metric_logger = kwargs['metric_logger']

    epoch_info = lgrs.loguru_logger.epoch_info()

    info_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    infos = [f'{k}: {v:>.4}' for k, v in info_dict.items()]

    # infos = [f'{k}: {meter.global_avg:>.4}' for k, meter in metric_logger.meters.items()]
    msg1 = ' '.join((epoch_info, 'Epoch Gathered Averaged Stats:', *infos))
    lgrs.loguru.success(msg1)

    if _is_main_process():
        run_state = lgrs.loguru_logger._get_run_state()
        if 'task_eval' in run_state:
            level = 'task'
            step = lgrs.loguru_logger._get_task_step()
        else:
            level = 'epoch'
            step = lgrs.loguru_logger._get_epoch_step()

        for k, v in info_dict.items():
            lgrs.neptune.neprun[f'eval/{level}/{k}'].append(v, step=step)


def eval_batch_start(**kwargs):
    # batch_info = lgrs.loguru_logger.batch_info()
    # samples = kwargs['samples']

    # msg1 = ' '.join((batch_info, f'Batch size is {samples.shape}'))
    # lgrs.loguru.info(msg1)
    pass

def eval_batch_end_with_interval(interval=10, **kwargs):
    batch_num = lgrs.loguru_logger.state['batch_num']
    num_batches = lgrs.loguru_logger.state['num_batches']
    if batch_num % interval == 0 or batch_num == num_batches - 1:
        batch_info = lgrs.loguru_logger.batch_info()
        infos = [f'{k}: {v:.4}' for k, v in kwargs.items()]
        msg1 = ' '.join((batch_info, *infos))
        lgrs.loguru.info(msg1)

        # if _is_main_process():
        #     for k, v in kwargs.items():
        #         lgrs.neptune.neprun[f'eval/batch/{k}'].append(v)


def task_start(**kwargs):
    task_info = lgrs.loguru_logger.task_info()
    msg1 = ' '.join((task_info, 'task start.'))
    lgrs.loguru.info(msg1)

    lgrs.loguru_logger.state['task_start_time'] = time.time()

def task_model_updated(**kwargs):
    model_without_ddp = kwargs['model_without_ddp']
    parameter_count = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)

    task_info = lgrs.loguru_logger.task_info()
    msg1 = ' '.join((task_info, f'number of parameters: {parameter_count} '))

    lgrs.loguru.info(msg1)

    if _is_main_process():
        step = lgrs.loguru_logger._get_task_step()
        lgrs.neptune.neprun[f'model/parameter_count'].append(parameter_count, step=step)


def task_end(**kwargs):
    task_time = time.time() - lgrs.loguru_logger.state['task_start_time']
    if _is_main_process():
        lgrs.neptune.neprun['time/task'].append(task_time)
    
    task_info = lgrs.loguru_logger.task_info()
    msg1 = ' '.join((task_info, f'task end. total time: {task_time:>.4}s'))
    lgrs.loguru.info(msg1)


def finetuning_end_checkpoint(**kwargs):
    if _is_checkpoint_enabled() and _is_main_process():
        lgrs.checkpoint.save_to_disk(**kwargs)
        task_info = lgrs.loguru_logger.task_info()
        msg1 = ' '.join((task_info, 'checkpoint saved!'))
        lgrs.loguru.info(msg1)

def train_end_save_memory(**kwargs):
    if _is_checkpoint_enabled() and _is_main_process():
        memory = kwargs['memory']
        filename = 'memory_' + lgrs.checkpoint._get_ckpt_info() + '.npz'
        path = lgrs.checkpoint.ckpt_dir / filename
        memory.save(path)

def train_end_load_memory(**kwargs):
    memory = kwargs['memory']

    filename = 'memory_' + lgrs.checkpoint._get_ckpt_info() + '.npz'
    path = lgrs.checkpoint.ckpt_dir / filename
    memory.load(path)

def main_start(**kwargs):
    node_info = lgrs.loguru_logger.node_info()

    msg1 = ' '.join((node_info, 'program start.'))
    lgrs.loguru.info(msg1)

    import torch, torchvision
    msg2 = f'torch {torch.__version__}, cuda {torch.version.cuda}, torchvision {torchvision.__version__}.'
    lgrs.loguru.info(msg2)

    params = kwargs.get('params')
    msg3 = ' '.join((node_info, str(params)))
    lgrs.loguru.info(msg3)

    if _is_main_process():
        lgrs.neptune.neprun['parameters'] = params

def main_end():
    node_info = lgrs.loguru_logger.node_info()

    if _is_main_process():
        lgrs.neptune.log_run_state()
        lgrs.neptune.neprun.stop()

    lgrs.save.save_to_disk()

    msg1 = ' '.join((node_info, 'program end.'))
    lgrs.loguru.info(msg1)

def _get_rc():
    return lgrs.loguru_logger.rc

def _is_main_process():
    rank = _get_rc().get('rank', 0)
    return rank == 0

def _is_checkpoint_enabled():
    return _get_rc().get('enable_checkpoint', False)
    