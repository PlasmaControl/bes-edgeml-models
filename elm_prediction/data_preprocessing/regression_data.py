"""
Data class to package BES data for training with regression algorithm.
changes target variable form class based to time based.
"""
from typing import Tuple

import numpy as np
import h5py

try:
    from .base_data import BaseData
except ImportError:
    from base_data import BaseData

