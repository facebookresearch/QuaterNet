# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from common.pose_network import PoseNetwork
from common.quaternion import qeuler, qeuler_np, qfix, euler_to_quaternion
import numpy as np
import torch

class PoseNetworkShortTerm(PoseNetwork):
    
    def __init__(self, prefix_length):
        super().__init__(prefix_length, 32, 0, 0, model_velocities=True)
        
    def _prepare_next_batch_impl(self, batch_size, dataset, target_length, sequences):
        super()._prepare_next_batch_impl(batch_size, dataset, target_length, sequences)
        
        buffer_quat = np.zeros((batch_size, self.prefix_length+target_length, 32*4), dtype='float32')
        buffer_euler = np.zeros((batch_size, target_length, 32*3), dtype='float32')
        
        sequences = np.random.permutation(sequences)

        batch_idx = 0
        for i, (subject, action) in enumerate(sequences):
            # Pick a random chunk from each sequence
            start_idx = np.random.randint(0, dataset[subject][action]['rotations'].shape[0] - self.prefix_length - target_length + 1)
            mid_idx = start_idx + self.prefix_length
            end_idx = start_idx + self.prefix_length + target_length
            
            buffer_quat[batch_idx] = dataset[subject][action]['rotations'][start_idx:end_idx].reshape( \
                                          self.prefix_length+target_length, -1)
            buffer_euler[batch_idx] = dataset[subject][action]['rotations_euler'][mid_idx:end_idx].reshape( \
                                          target_length, -1)
            
            batch_idx += 1
            if batch_idx == batch_size or i == len(sequences) - 1:
                yield buffer_quat[:batch_idx], buffer_euler[:batch_idx]
                batch_idx = 0
                
    def _loss_impl(self, predicted, expected):
        """
        Euler angle loss.
        - predicted is given as quaternions.
        - expected is given as zyx-ordered Euler angles.
        """
        super()._loss_impl(predicted, expected)
        
        predicted_quat = predicted.view(predicted.shape[0], predicted.shape[1], -1, 4)
        expected_euler = expected.view(predicted.shape[0], predicted.shape[1], -1, 3)
        
        predicted_euler = qeuler(predicted_quat, order='zyx', epsilon=1e-6)
        # L1 loss on angle distance with 2pi wrap-around
        angle_distance = torch.remainder(predicted_euler - expected_euler + np.pi, 2*np.pi) - np.pi
        return torch.mean(torch.abs(angle_distance))
    
    def predict(self, prefix, target_length):
        """
        Predict a sequence using the given prefix.
        """
        assert target_length > 0
        
        with torch.no_grad():
            prefix = prefix.reshape(prefix.shape[1], -1, 4)
            prefix = qeuler_np(prefix, 'zyx')
            prefix = qfix(euler_to_quaternion(prefix, 'zyx'))
            inputs = torch.from_numpy(prefix.reshape(1, prefix.shape[0], -1).astype('float32'))
            
            if self.use_cuda:
                inputs = inputs.cuda()

            predicted, hidden = self.model(inputs)
            frames = [predicted]

            for i in range(1, target_length):
                predicted, hidden = self.model(predicted, hidden)
                frames.append(predicted)

            result = torch.cat(frames, dim=1)
            return result.view(result.shape[0], result.shape[1], -1, 4).cpu().numpy()