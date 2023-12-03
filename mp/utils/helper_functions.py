# ------------------------------------------------------------------------------
# Miscellaneous helper functions.
# ------------------------------------------------------------------------------

import datetime
import numpy as np
import random
import torch


def get_time_string(cover=False):
    r"""
    Returns the current time in the format YYYY-MM-DD_HH-MM, or
    [YYYY-MM-DD_HH-MM] if 'cover' is set to 'True'.
    """
    date = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0]
    if cover:
        return '[' + date + ']'
    else:
        return date


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
