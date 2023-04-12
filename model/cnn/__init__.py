from .abstract import AbstractCNN
from .inception import InceptionV3
from .senet import legacy_seresnet18 as seresnet18
from .resnet import (
    resnet18, resnet34, resnet50, resnext50_32x4d, wide_resnet50_2
)
from .resnet_scs import resnet18_scs, resnet18_scs_avg, resnet18_scs_max
from .vgg import vgg16_bn, vgg16
from .resnet_rebuffi import CifarResNet as rebuffi
