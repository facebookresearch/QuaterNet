# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from long_term.pace_network import PaceNetwork
from long_term.dataset_locomotion import dataset, actions_valid
from long_term.locomotion_utils import build_extra_features, compute_splines
torch.manual_seed(1234)

if __name__ == '__main__':
    dataset.compute_positions()
    build_extra_features(dataset)
    compute_splines(dataset)

    model = PaceNetwork()
    if torch.cuda.is_available():
        model.cuda()

    chunk_length = 1000
    batch_size = 40
    sequences_train = []
    sequences_valid = []
    n_discarded = 0
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            if 'sidestep' in action:
                # We don't really want those
                continue

            if dataset[subject][action]['spline'].size() < chunk_length:
                n_discarded += 1
                continue

            train = True
            for action_valid in actions_valid:
                if action.startswith(action_valid):
                    train = False
                    break
            if train:
                sequences_train.append((subject, action))
            else:
                sequences_valid.append((subject, action))

    print('%d sequences were discarded for being too short.' % n_discarded)
    print('Training on %d sequences, validating on %d sequences.' % (len(sequences_train), len(sequences_valid)))
    model.train(dataset, sequences_train, sequences_valid, batch_size, chunk_length)
    model.save_weights('weights_pace_network.bin')