import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EKF_ROOT = os.path.dirname(HERE)
IMU_ROOT = os.path.join(os.path.dirname(EKF_ROOT), "IMU")
for p in (EKF_ROOT, IMU_ROOT, os.path.join(IMU_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)
