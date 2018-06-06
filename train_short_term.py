# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from short_term.pose_network_short_term import PoseNetworkShortTerm
from short_term.dataset_h36m import dataset, subjects_train, subjects_valid, short_term_weights_path
from test_short_term import run_evaluation
torch.manual_seed(1234)

if __name__ == '__main__':
    model = PoseNetworkShortTerm(prefix_length=50)
    if torch.cuda.is_available():
        model.cuda()

    sequences_train = []
    sequences_valid = []
    for subject in subjects_train:
        for action in dataset[subject].keys():
            sequences_train.append((subject, action))
    for subject in subjects_valid:
        for action in dataset[subject].keys():
            sequences_valid.append((subject, action))

    print('Training on %d sequences, validating on %d sequences.' % (len(sequences_train), len(sequences_valid)))

    target_length = 10
    dataset.compute_euler_angles(order='zyx')
    model.train(dataset, target_length, sequences_train, sequences_valid, batch_size=60, n_epochs=3000)
    model.save_weights(short_term_weights_path)
    model.eval()
    with torch.no_grad():
        run_evaluation(model)