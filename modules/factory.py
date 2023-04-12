import torch

from . import utils, samplers
from model import convit, convit_lpa
from model.cnn import (InceptionV3, rebuffi, resnet18, resnet34, resnet50, resnext50_32x4d, seresnet18, vgg16, vgg16_bn, wide_resnet50_2, resnet18_scs, resnet18_scs_max, resnet18_scs_avg)
from continual import dytox, raw


def get_backbone(args):
    model_name = args.model
    if model_name == 'convit':
        model = convit.VisionTransformer(
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            img_size=args.input_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            local_up_to_layer=args.local_up_to_layer,
            locality_strength=args.locality_strength
        )
    elif model_name == 'convit_lpa':
        model = convit_lpa.VisionTransformer(
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            img_size=args.input_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            local_up_to_layer=args.local_up_to_layer,
            locality_strength=args.locality_strength,
            num_lpsa=args.num_lpsa, 
            hyper_lambda=args.hyper_lambda
        )
    elif model_name == 'resnet18_scs': model = resnet18_scs()
    elif model_name == 'resnet18_scs_avg': model = resnet18_scs_max()
    elif model_name == 'resnet18_scs_max': model = resnet18_scs_avg()
    elif model_name == 'resnet18': model = resnet18()
    elif model_name == 'resnet34': model = resnet34()
    elif model_name == 'resnet50': model = resnet50()
    elif model_name == 'wide_resnet50': model = wide_resnet50_2()
    elif model_name == 'resnext50': model = resnext50_32x4d()
    elif model_name == 'seresnet18': model = seresnet18()
    elif model_name == 'inception3': model = InceptionV3()
    elif model_name == 'vgg16bn': model = vgg16_bn()
    elif model_name == 'vgg16': model = vgg16()
    elif model_name == 'rebuffi': model = rebuffi()
    else:
        raise NotImplementedError(f'Unknown backbone {model_name}')

    return model



def get_loaders(dataset_train, dataset_val, args):
    sampler_train, sampler_val = samplers.get_sampler(dataset_train, dataset_val, args)

    actual_batch_size = args.batch_size // utils.get_world_size() * 2 # default 2 gpus and 128*2 batchsize
    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=actual_batch_size, 
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True
    )

    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * actual_batch_size), 
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True
    )

    return loader_train, loader_val


def get_train_loaders(dataset_train, args):
    sampler_train = samplers.get_train_sampler(dataset_train, args)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.replay_memory if args.replay_memory > 0 else args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    return loader_train



class InfiniteLoader:
    def __init__(self, loader):
        self.loader = loader
        self.reset()

    def reset(self):
        self.it = iter(self.loader)

    def get(self):
        try:
            return next(self.it)
        except StopIteration:
            self.reset()
            return self.get()


def update_model(model_without_ddp, task_id, args):
    if args.method == 'dytox':
        return update_dytox(model_without_ddp, task_id, args)
    elif args.method == 'raw':
        return update_raw(model_without_ddp, task_id, args)
    else:
        raise ValueError(f'Unknown method name <{args.method}>.')


def update_dytox(model_without_ddp, task_id, args):
    if task_id == 0:
        model_without_ddp = dytox.DyTox(
            model_without_ddp,
            nb_classes=args.initial_increment,
            individual_classifier=args.ind_clf,
            head_div=args.head_div > 0.,
            head_div_mode=args.head_div_mode,
            joint_tokens=args.joint_tokens
        )
    else:
        model_without_ddp.update_model(args.increment)

    return model_without_ddp

def update_raw(model_without_ddp, task_id, args):
    if task_id == 0:
        model_without_ddp = raw.Raw(
            model_without_ddp,
            nb_classes=args.initial_increment,
            individual_classifier=args.ind_clf
        )
    else:
        model_without_ddp.update_model(args.increment)

    return model_without_ddp