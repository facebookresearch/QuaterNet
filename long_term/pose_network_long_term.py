# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from common.pose_network import PoseNetwork
from common.quaternion import qeuler, qmul_np, qrot_np
from time import time
import numpy as np
import torch

class PoseNetworkLongTerm(PoseNetwork):
    
    def __init__(self, prefix_length, skeleton):
        super().__init__(prefix_length, skeleton.num_joints(), 2, 6, model_velocities=False)
        self.skeleton = skeleton
        self.zero_buf = torch.zeros(1)
        
    def cuda(self):
        super().cuda()
        self.zero_buf = self.zero_buf.cuda()
        self.skeleton.cuda()
        return self
    
    def _rotate_batch(self, buffer_rot, buffer_pos):
        """
        Data augmentation: rotate each sequence in this batch by a random angle.
        """
        # We cover a 4*pi range to make sure that we visit the full quaternion range
        random_angles = np.random.uniform(-2*np.pi, 2*np.pi, size=buffer_rot.shape[0])
        rrot = np.zeros((random_angles.shape[0], 4), dtype='float32')
        rrot[:, 0] = np.cos(random_angles/2)
        rrot[:, 2] = np.sin(random_angles/2)
        
        # Rotate root joints
        rrot_ = np.tile(rrot.reshape(-1, 1, 4), (1, buffer_rot.shape[1], 1))
        buffer_rot[:, :, :4] = qmul_np(rrot_, buffer_rot[:, :, :4])
        
        # Rotate positions
        nj = self.skeleton.num_joints()
        positions = buffer_pos[:, :, :nj*3].reshape(buffer_pos.shape[0], buffer_pos.shape[1], -1, 3)
        rrot_ = np.tile(rrot.reshape(-1, 1, 1, 4), (1, positions.shape[1], positions.shape[2], 1))
        buffer_pos[:, :, :nj*3] = qrot_np(rrot_, positions).reshape(buffer_pos[:, :, :nj*3].shape)
        
        # Rotate controls
        extra_angles = np.zeros((buffer_rot.shape[0], buffer_rot.shape[1], 2, 3), dtype='float32')
        extra_angles[:, :, 0, [0, 2]] = buffer_rot[:, :, [-2, -1]]
        extra_angles[:, :, 1, [0, 2]] = buffer_rot[:, :, [-4, -3]]
        rrot_ = np.tile(rrot.reshape(-1, 1, 1, 4), (1, extra_angles.shape[1], extra_angles.shape[2], 1))
        extra_angles = qrot_np(rrot_, extra_angles)
        buffer_rot[:, :, [-2, -1]] = extra_angles[:, :, 0, [0, 2]]
        buffer_rot[:, :, [-4, -3]] = extra_angles[:, :, 1, [0, 2]]
        
        return buffer_rot, buffer_pos
    
    def _prepare_next_batch_impl(self, batch_size, dataset, target_length, sequences):
        super()._prepare_next_batch_impl(batch_size, dataset, target_length, sequences)
        
        assert dataset.skeleton() == self.skeleton
        nj = self.skeleton.num_joints()
        
        # The memory layout of the batches is: rotations or positions | translations | controls
        buffer_rot = np.zeros((batch_size, self.prefix_length+target_length,
                               nj*4 + self.translations_size + self.controls_size), dtype='float32')
        buffer_pos = np.zeros((batch_size, target_length, nj*3 + self.translations_size), dtype='float32')
        
        probs = []
        for i, (subject, action) in enumerate(sequences):
            probs.append(dataset[subject][action]['rotations'].shape[0])
        probs = np.array(probs)/np.sum(probs)
        
        pseudo_passes = (len(sequences) + batch_size - 1) // batch_size # Round in excess
        for p in range(pseudo_passes):
            idxs = np.random.choice(len(sequences), size=batch_size, replace=True, p=probs)
            for i, (subject, action) in enumerate(np.array(sequences)[idxs]):
                # Pick a random chunk
                full_seq_length = dataset[subject][action]['rotations'].shape[0]
                max_index = full_seq_length - self.prefix_length - target_length + 1
                start_idx = np.random.randint(0, max_index)
                mid_idx = start_idx + self.prefix_length
                end_idx = start_idx + self.prefix_length + target_length

                buffer_rot[i, :, :nj*4] = dataset[subject][action]['rotations'][start_idx:end_idx].reshape( \
                                              self.prefix_length+target_length, -1)
                buffer_rot[i, :, nj*4:] = dataset[subject][action]['extra_features'][start_idx:end_idx]

                buffer_pos[i, :, :nj*3] = dataset[subject][action]['positions_local'][mid_idx:end_idx].reshape( \
                                              target_length, -1)
                buffer_pos[i, :, nj*3:] = dataset[subject][action]['extra_features'][mid_idx:end_idx, :self.translations_size]
                        
            # Perform data augmentation
            buffer_rot[:], buffer_pos[:] = self._rotate_batch(buffer_rot, buffer_pos)

            yield buffer_rot, buffer_pos
                
    def _loss_impl(self, predicted, expected):
        """
        Positional loss with forward kinematics.
        - predicted is given as quaternions.
        - expected is given as 3D positions.
        """
        super()._loss_impl(predicted, expected)
        
        nj = self.skeleton.num_joints()
        predicted_extra = predicted[:, :, nj*4 : nj*4+self.translations_size]
        expected_extra = expected[:, :, nj*3:]

        predicted_rotations = predicted[:, :, :nj*4].view(predicted.shape[0], predicted.shape[1], -1, 4)
        predicted_positions = self.skeleton.forward_kinematics(predicted_rotations,
                                    self.zero_buf.expand(predicted_rotations.shape[0], predicted_rotations.shape[1], 3))
        
        # Add height
        predicted_positions[:, :, :, 1] += predicted_extra[:, :, 1].unsqueeze(2).expand(-1, -1, nj)

        # Euclidean distance (with joint-wise square root, not MSE!)
        expected_positions = expected[:, :, :nj*3].view(predicted_positions.shape)
        differences = (predicted_positions - expected_positions).norm(dim=3)
        
        # Longitudinal offset w.r.t. trajectory
        displacements = torch.abs(predicted_extra[:, :, 0] - expected_extra[:, :, 0]) \
            .view(expected_extra.shape[0], expected_extra.shape[1], 1)
        
        # We apply the longitudinal offset to the loss function by penalizing all joints.
        # This is equivalent to multiplying the "displacements" loss by the number of joints.
        return torch.mean(differences + displacements.expand_as(differences))
    
    def generate_motion(self, spline, action_prefix):
        """
        Generate a locomotion sequence using this model.
        Arguments:
         -- spline: the path to follow (an object of type Spline), annotated with all extra inputs (e.g. local speed).
         -- action_prefix: the action which will be used to initialize the model for the first "prefix_length" frames.
        """
        with torch.no_grad():
            start_time = time()
            rot0 = action_prefix['rotations'][:, :self.prefix_length].reshape(-1, self.skeleton.num_joints()*4)
            extra0 = action_prefix['extra_features'][:, :self.prefix_length]
            phase0 = np.arctan2(extra0[-1, 3], extra0[-1, 2])
            
            inputs = torch.from_numpy(np.concatenate((rot0, extra0), axis=1).astype('float32')).unsqueeze(0)
            if self.use_cuda:
                inputs = inputs.cuda()

            predicted, hidden = self.model(inputs)
            
            def next_frame(phase, amplitude, direction, facing_direction):
                nonlocal predicted, hidden
                features = torch.Tensor([np.cos(phase)*amplitude, np.sin(phase)*amplitude,
                                         direction[0]*amplitude, direction[1]*amplitude,
                                         facing_direction[0], facing_direction[1]]).view(1, 1, -1)
                if self.use_cuda:
                    features = features.cuda()
                predicted, hidden = self.model(torch.cat((predicted, features), dim=2), hidden)
                return predicted[:, :, :self.skeleton.num_joints()*4].cpu().numpy().reshape(-1, 4), \
                       predicted[0, 0, -2].item(), predicted[0, 0, -1].item() # Spline offset and height

            current_phase = phase0

            travel_distance = 0 # Longitudinal distance traveled along the spline
            rotations = [] # Joint rotations at each time step
            positions = [] # Root joint positions at each time step
            speed = 0
            stop = False

            while not stop:
                direction = spline.interpolate(travel_distance, 'tangent')
                facing_direction = spline.interpolate(travel_distance, 'direction')
                avg_speed = spline.interpolate(travel_distance, 'amplitude').squeeze()
                freq = spline.interpolate(travel_distance, 'frequency').squeeze()

                rot, displacement, height = next_frame(current_phase, avg_speed, direction, facing_direction)

                current_phase = (current_phase + freq) % (2*np.pi)
                travel_distance += avg_speed
                next_distance = travel_distance + displacement

                if next_distance < spline.length():
                    rotations.append(rot)
                    positions.append((next_distance, height))
                else:
                    # End of spline reached
                    stop = True

            out_trajectory = torch.FloatTensor(len(positions), 3)
            distances, heights = zip(*positions)
            out_trajectory[:, [0, 2]] = torch.from_numpy(spline.interpolate(distances).astype('float32'))
            out_trajectory[:, 1] = torch.FloatTensor(list(heights))
            out_trajectory = out_trajectory.unsqueeze(0)
            rotations = torch.FloatTensor(rotations).unsqueeze(0)
            
            if self.use_cuda:
                out_trajectory = out_trajectory.cuda()
                rotations = rotations.cuda()

            result = self.skeleton.forward_kinematics(rotations, out_trajectory).squeeze(0).cpu().numpy()
            end_time = time()
            duration = end_time - start_time
            fps = result.shape[0] / duration
            print('%d frames generated in %.3f seconds (%.2f FPS)' % (result.shape[0], duration, fps))
            return result