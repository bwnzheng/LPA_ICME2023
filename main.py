import os
import copy
import datetime

import torch
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from continuum.tasks import split_train_val

from continual import scaler, rehearsal, mixup
from modules import utils, datasets, factory, engine, losses
import logger as lgr
from logger import loggers as lgrs

def main(args):
    lgr.update_state(run_state='prepare')
    lgr.main_start(params=vars(args))

    use_distillation = args.auto_kd
    device = torch.device(args.device)

    # fix the seed for reproducibility
    utils.set_seed(args.seed)

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    scenario_train, args.nb_classes = datasets.build_dataset(is_train=True, args=args)
    scenario_val, _ = datasets.build_dataset(is_train=False, args=args)
    scenario_rehearsal, _ = datasets.build_dataset(is_train=True, args=args, noaug=True)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None

    model = factory.get_backbone(args)
    model.to(device)
    # model will be on multiple GPUs, while model_without_ddp on a single GPU, but
    # it's actually the same model.
    model_without_ddp = model
    lgr.prepare_backbone(backbone_model=model_without_ddp)
    
    if args.distributed:
        torch.distributed.barrier()

    loss_scaler = scaler.ContinualScaler(args.no_amp)

    criterion = losses.select_criterion(args)
    # lgrs.loguru.debug('criterion: ' + str(criterion))

    teacher_model = None

    memory = None
    if args.memory_size > 0:
        memory = rehearsal.Memory(
            args.memory_size, scenario_train.nb_classes, args.rehearsal, args.fixed_memory)

    base_lr = args.lr

    if args.debug:
        args.base_epochs = 1
        args.epochs = 1

    # --------------------------------------------------------------------------
    #
    # Begin of the task loop
    #
    # --------------------------------------------------------------------------
    dataset_true_val = None

    lgr.update_state(num_tasks=len(scenario_train))
    for task_id, dataset_train in enumerate(scenario_train):
        if args.max_task == task_id:
            break
        lgr.update_state(task_num=task_id)
        lgr.task_start()

        # ----------------------------------------------------------------------
        # Data
        dataset_val = scenario_val[:task_id + 1]
        if args.validation > 0.:  # use validation split instead of test
            if task_id == 0:
                dataset_train, dataset_val = split_train_val(dataset_train, args.validation)
                dataset_true_val = dataset_val
            else:
                dataset_train, dataset_val = split_train_val(dataset_train, args.validation)
                dataset_true_val.concat(dataset_val)
            dataset_val = dataset_true_val

        for i in range(3):  # Quick check to ensure same preprocessing between train/test
            assert abs(dataset_train.trsf.transforms[-1].mean[i] - dataset_val.trsf.transforms[-1].mean[i]) < 0.0001
            assert abs(dataset_train.trsf.transforms[-1].std[i] - dataset_val.trsf.transforms[-1].std[i]) < 0.0001

        loader_memory = None
        if task_id > 0 and memory is not None and not args.retrain_scratch:
            dataset_memory = memory.get_dataset(dataset_train)
            loader_memory = factory.InfiniteLoader(factory.get_train_loaders(
                dataset_memory, args
            ))
            if not args.sep_memory:
                for _ in range(args.oversample_memory):
                    dataset_train.add_samples(*memory.get())

            if args.only_ft:
                dataset_train = rehearsal.get_finetuning_dataset(dataset_train, memory, 'balanced')
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Initializing teacher model from previous task
        if use_distillation and task_id > 0 and not args.retrain_scratch:
            teacher_model = copy.deepcopy(model_without_ddp)
            teacher_model.freeze(['all'])
            teacher_model.eval()
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Ensembling
        model_without_ddp = factory.update_model(model_without_ddp, task_id, args)
        # ----------------------------------------------------------------------
        utils.set_parameters(model_without_ddp, requires_grad=True)
        lgr.task_model_updated(model_without_ddp=model_without_ddp)

        # ----------------------------------------------------------------------
        # Debug: Joint training from scratch on all previous data
        if args.retrain_scratch:
            model_without_ddp.reset_parameters()
            dataset_train = scenario_train[:task_id+1]
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        if task_id > 0 and not args.retrain_scratch:
            model_without_ddp.freeze(args.freeze_task)
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        # Data
        loader_train, loader_val = factory.get_loaders(dataset_train, dataset_val, args)
        # ----------------------------------------------------------------------

        # ----------------------------------------------------------------------
        if task_id > 0 and args.retrain_scratch and args.incremental_batch_size:
            args.batch_size = args.incremental_batch_size

        # Learning rate and optimizer
        if args.incremental_lr is not None and task_id > 0 and not args.retrain_scratch:
            linear_scaled_lr = args.incremental_lr / 2
        else:
            linear_scaled_lr = base_lr / 2

        args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, model_without_ddp)
        lr_scheduler, _ = create_scheduler(args, optimizer)
        # ----------------------------------------------------------------------

        if mixup_active:
            mixup_fn = mixup.Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=dataset_val.nb_classes,
                loader_memory=loader_memory,
                patch_size=args.patch_size
            )

        skipped_task = False
        initial_epoch = epoch = 0
        if args.auto_resume:
            skipped_task = lgrs.checkpoint.load_from_disk(
                model=model_without_ddp,
                # optimizer=optimizer,
                # lr_scheduler=lr_scheduler,
                loss_scaler=loss_scaler
                )
        
        if skipped_task:
            epochs = 0
        elif args.base_epochs is not None and task_id == 0:
            epochs = args.base_epochs
        else:
            epochs = args.epochs

        if args.distributed:
            del model
            model = torch.nn.parallel.DistributedDataParallel(
                model_without_ddp, device_ids=[args.rank], find_unused_parameters=args.debug)
            torch.distributed.barrier()
        else:
            model = model_without_ddp

        model_without_ddp.nb_epochs = epochs
        model_without_ddp.nb_batch_per_epoch = len(loader_train)

        lgr.update_state(num_epochs=epochs)
        for epoch in range(initial_epoch, epochs):
            lgr.update_state(epoch_num=epoch, run_state='train')
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            engine.train_one_epoch(
                model, loader_train, criterion, optimizer, device, task_id, loss_scaler, args, mixup_fn, teacher_model, model_without_ddp
            )

            lr_scheduler.step(epoch)

            if args.eval_interval and (epoch % args.eval_interval  == 0 or epoch == epochs - 1):
                lgr.update_state(run_state='eval')
                engine.evaluate(loader_val, model, task_id, device)

        # update memory
        if memory is not None:
            if skipped_task:
                lgr.train_end_load_memory(memory=memory)
            else:
                task_set_to_rehearse = scenario_rehearsal[task_id]

                memory.add(task_set_to_rehearse, model, args.initial_increment if task_id == 0 else args.increment)

                lgr.train_end_save_memory(memory=memory)

            assert len(memory) <= args.memory_size
            torch.distributed.barrier()


        # ----------------------------------------------------------------------
        # FINETUNING
        # ----------------------------------------------------------------------
        lgr.update_state(run_state='finetune')
        if args.finetuning and memory and (task_id > 0 or scenario_train.nb_classes == args.initial_increment) and not skipped_task and not args.retrain_scratch:
            dataset_finetune = rehearsal.get_finetuning_dataset(dataset_train, memory, args.finetuning)

            loader_finetune, loader_val = factory.get_loaders(dataset_finetune, dataset_val, args)
            if args.finetuning_resetclf:
                model_without_ddp.reset_classifier()

            model_without_ddp.freeze(args.freeze_ft)

            if args.distributed:
                del model
                model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.rank])
                torch.distributed.barrier()
            else:
                model = model_without_ddp

            model_without_ddp.begin_finetuning()

            args.lr  = args.finetuning_lr
            optimizer = create_optimizer(args, model_without_ddp)

            lgr.update_state(num_epochs=args.finetuning_epochs)
            for epoch in range(args.finetuning_epochs):
                lgr.update_state(run_state='finetune', epoch_num=epoch)
                if args.distributed and hasattr(loader_finetune.sampler, 'set_epoch'):
                    loader_finetune.sampler.set_epoch(epoch)
                
                engine.train_one_epoch(model, loader_finetune, criterion, optimizer, device, task_id, loss_scaler, args, mixup_fn, teacher_model, model_without_ddp)

                if epoch % 10 == 0 or epoch == args.finetuning_epochs - 1:
                    lgr.update_state(run_state='eval')
                    engine.evaluate(loader_val, model, task_id, device)

            model_without_ddp.end_finetuning()
        
        lgr.finetuning_end_checkpoint(
            model=model_without_ddp.state_dict(),
            # optimizer=optimizer.state_dict(),
            # lr_scheduler=lr_scheduler.state_dict(),
            loss_scaler=loss_scaler.state_dict()
            )

        lgr.update_state(run_state='task_eval')
        engine.evaluate(loader_val, model, task_id, device)
        lgr.task_end()
    
    lgr.update_state(run_state='finished')
    lgr.main_end()


def init_torch_dist(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = True
    else:
        args.distributed = False
        return

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend=args.dist_backend, 
        world_size=args.world_size, 
        rank=args.rank,
        timeout=datetime.timedelta(seconds=5400))
    torch.distributed.barrier()


if __name__ == '__main__':
    # get args
    from modules.arguments import get_args
    args = get_args()

    # init torch dist
    init_torch_dist(args)

    # init rc
    lgr.init_rc(args)
    
    # init loggers
    lgrs.init_loguru()
    lgrs.init_neptune()
    lgrs.init_checkpoint()

    with lgrs.loguru.catch():
        main(args)

