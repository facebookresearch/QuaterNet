# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from long_term.pose_network_long_term import PoseNetworkLongTerm
from long_term.dataset_locomotion import dataset, actions_valid, long_term_weights_path
from long_term.locomotion_utils import build_extra_features
torch.manual_seed(1234)

if __name__ == '__main__':
    prefix_length = 30
    target_length = 60

    model = PoseNetworkLongTerm(prefix_length, dataset.skeleton())
    if torch.cuda.is_available():
        model.cuda()
        dataset.cuda()

    sequences_train = []
    sequences_valid = []
    n_discarded = 0
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            if dataset[subject][action]['rotations'].shape[0] < prefix_length + target_length:
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
    dataset.compute_positions()
    build_extra_features(dataset)
    model.train(dataset, target_length, sequences_train, sequences_valid, batch_size=40, n_epochs=4000)
    model.save_weights(long_term_weights_path)