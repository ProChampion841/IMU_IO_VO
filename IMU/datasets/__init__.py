"""Dataset package.  UAV is the ONLY loader.

EuRoC, KITTI, SubT and TUM-VI were removed on 2026-09-07: none of them had been
used on this project, all four carried their own conventions into shared code
(see the `mti_or_gt` / `airspeed_or_zeros` fallbacks they justified), and KITTI
needed an optional third-party package just to import.  `Sequence.subclasses`
now resolves exactly one name, "UAV", which is what every config in configs/
asks for.
"""
from .dataset import *
from .dataset_utils import *
from .UAVdataset import *
