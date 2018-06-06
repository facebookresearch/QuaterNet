# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import sys
import matplotlib.pyplot as plt
from long_term.dataset_locomotion import dataset, long_term_weights_path
from long_term.locomotion_utils import build_extra_features, compute_splines
from long_term.pose_network_long_term import PoseNetworkLongTerm
from common.spline import Spline
from common.visualization import render_animation
from long_term.pace_network import PaceNetwork

default_subject = 'S1'

if __name__ == '__main__':
    if len(sys.argv) > 3:
        raise ValueError("Invalid number of arguments")
    elif len(sys.argv) == 2 and sys.argv[1] in ['--list', '-l']:
        for action in dataset[default_subject].keys():
            print(action)
        exit()
        
    if torch.cuda.is_available():
        print('CUDA detected. Using GPU.')
        dataset.cuda()
    else:
        print('CUDA not detected. Using CPU.')
    dataset.compute_positions()
    build_extra_features(dataset)
    compute_splines(dataset)

    pace_net = PaceNetwork()
    pace_net.load_weights('weights_pace_network.bin')
    
    model = PoseNetworkLongTerm(30, dataset.skeleton())
    if torch.cuda.is_available():
        model.cuda()
    model.load_weights(long_term_weights_path) # Load pretrained model
    
    if len(sys.argv) == 1:
        for subject in dataset.subjects():
            for action in dataset[subject].keys():
                if '_d0' not in action or '_m' in action:
                    continue
                print('Showing subject %s, action %s.' % (subject, action))
                annotated_spline = pace_net.predict(dataset[subject][action]['spline'])
                animation = model.generate_motion(annotated_spline, dataset[subject][action])
                render_animation(animation, dataset.skeleton(), dataset.fps(), output='interactive')
    else:
        # Visualize a particular action
        action = sys.argv[1]
        if action not in dataset[default_subject].keys():
            raise ValueError("The specified animation does not exist")
        annotated_spline = pace_net.predict(dataset[default_subject][action]['spline'])
        animation = model.generate_motion(annotated_spline, dataset[default_subject][action])
        if len(sys.argv) == 2:
            output_mode = 'interactive'
        else:
            plt.switch_backend('agg')
            output_mode = sys.argv[2]
        render_animation(animation, dataset.skeleton(), dataset.fps(), output=output_mode)
        
