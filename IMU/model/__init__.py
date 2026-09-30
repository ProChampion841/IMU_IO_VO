from .net import ModelBase
from .cnn import CNNPOS
from .others import Identity, ParamNet
from .code import *
from .hybrid import HybridNet
from .velocity_net import VelocityNet

net_dict = {
    'codeposenet': CodePoseNet,
    'iden': Identity,
    'cnnpos': CNNPOS,
    'codenet': CodeNet,
    'param': ParamNet,
    'hybridnet': HybridNet,
    'velnet': VelocityNet,
}
