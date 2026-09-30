"""Model package.  velnet is the ONLY network.

HybridNet (the IMU-CORRECTION arm) and the older CodeNet / CodePoseNet / CNNPOS /
Identity / ParamNet baselines were removed on 2026-09-07.  Their shared trunk
survives as `model/encoder.py` -- VelocityNet subclasses it -- and `model/cnn.py`
keeps CNNEncoder for the same reason.

Recover any of them with: git show <commit>^:model/<name>.py
"""
from .net import ModelBase
from .encoder import Encoder
from .velocity_net import VelocityNet

net_dict = {
    'velnet': VelocityNet,
}
