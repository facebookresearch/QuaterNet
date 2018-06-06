# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import re
import numpy as np
import pandas as pd
import os.path
import pickle
from glob import glob
from common.quaternion import expmap_to_quaternion, qfix
from common.mocap_dataset import MocapDataset
from common.skeleton import Skeleton

# Set to True for validation, set to False for testing
perform_validation = False

if perform_validation:
    subjects_train = ['S1', 'S7', 'S8', 'S9', 'S11']
    subjects_valid = ['S6']
    subjects_test = ['S5']
else:
    subjects_train = ['S1', 'S6', 'S7', 'S8', 'S9', 'S11']
    subjects_valid = []
    subjects_test = ['S5']

dataset_path = 'datasets/dataset_h36m.npz'
short_term_weights_path = 'weights_short_term.bin'

skeleton_h36m = Skeleton(offsets=[
       [   0.      ,    0.      ,    0.      ],
       [-132.948591,    0.      ,    0.      ],
       [   0.      , -442.894612,    0.      ],
       [   0.      , -454.206447,    0.      ],
       [   0.      ,    0.      ,  162.767078],
       [   0.      ,    0.      ,   74.999437],
       [ 132.948826,    0.      ,    0.      ],
       [   0.      , -442.894413,    0.      ],
       [   0.      , -454.20659 ,    0.      ],
       [   0.      ,    0.      ,  162.767426],
       [   0.      ,    0.      ,   74.999948],
       [   0.      ,    0.1     ,    0.      ],
       [   0.      ,  233.383263,    0.      ],
       [   0.      ,  257.077681,    0.      ],
       [   0.      ,  121.134938,    0.      ],
       [   0.      ,  115.002227,    0.      ],
       [   0.      ,  257.077681,    0.      ],
       [   0.      ,  151.034226,    0.      ],
       [   0.      ,  278.882773,    0.      ],
       [   0.      ,  251.733451,    0.      ],
       [   0.      ,    0.      ,    0.      ],
       [   0.      ,    0.      ,   99.999627],
       [   0.      ,  100.000188,    0.      ],
       [   0.      ,    0.      ,    0.      ],
       [   0.      ,  257.077681,    0.      ],
       [   0.      ,  151.031437,    0.      ],
       [   0.      ,  278.892924,    0.      ],
       [   0.      ,  251.72868 ,    0.      ],
       [   0.      ,    0.      ,    0.      ],
       [   0.      ,    0.      ,   99.999888],
       [   0.      ,  137.499922,    0.      ],
       [   0.      ,    0.      ,    0.      ]
    ],
    parents=[-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
       16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
    joints_left=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
    joints_right=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23])

dataset = MocapDataset(dataset_path, skeleton_h36m, fps=50)
dataset.downsample(2)