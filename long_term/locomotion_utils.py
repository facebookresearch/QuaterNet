# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import scipy.ndimage.filters
import numpy as np
from common.spline import Spline

lf = 4 # Left foot index
rf = 9 # Right foot index

def build_phase_track(positions_world):
    """
    Detect foot steps and extract a control signal that describes the current state of the walking cycle.
    This is based on the assumption that the speed of a foot is almost zero during a contact.
    """
    l_speed = np.linalg.norm(np.diff(positions_world[:, lf], axis=0), axis=1)
    r_speed = np.linalg.norm(np.diff(positions_world[:, rf], axis=0), axis=1)
    displacements = np.cumsum(np.linalg.norm(np.diff(positions_world[:, 0], axis=0), axis=1))
    left_contact = l_speed[0] < r_speed[0]
    epsilon = 0.1 # Hysteresis (i.e. minimum height difference before a foot switch is triggered)
    cooldown = 3 # Minimum # of frames between steps
    accumulator = np.pi if left_contact else 0
    phase_points = [ (0, accumulator) ]
    disp_points = [ (0, displacements[0]) ]
    i = cooldown
    while i < len(l_speed):
        if left_contact and l_speed[i] > r_speed[i] + epsilon:
            left_contact = False
            accumulator += np.pi
            phase_points.append((i, accumulator))
            disp_points.append((i, displacements[i] - displacements[disp_points[-1][0]]))
            i += cooldown
        elif not left_contact and r_speed[i] > l_speed[i] + epsilon:
            left_contact = True
            accumulator += np.pi
            phase_points.append((i, accumulator))
            disp_points.append((i, displacements[i] - displacements[disp_points[-1][0]]))
            i += cooldown
        else:
            i += 1
            
    phase = np.zeros(l_speed.shape[0])
    end_idx = 0
    for i in range(len(phase_points) - 1):
        start_idx = phase_points[i][0]
        end_idx = phase_points[i+1][0]
        phase[start_idx:end_idx] = np.linspace(phase_points[i][1], phase_points[i+1][1], end_idx-start_idx, endpoint=False)
    phase[end_idx:] = phase_points[-1][1]
    last_point = (phase[-1] - phase[-2]) + phase[-1]
    phase = np.concatenate((phase, [last_point]))
    return phase

def extract_translations_controls(positions_world, xy_orientation):
    """
    Extract extra features for the long-term network:
    - Translations: longitudinal speed along the spline; height of the root joint.
    - Controls: walking phase as a [cos(theta), sin(theta)] signal; same for the facing direction and movement direction.
    """
    phase = build_phase_track(positions_world)
    xy = np.diff(positions_world[:, 0, [0, 2]], axis=0)
    xy = np.concatenate((xy, xy[-1:]), axis=0)
    z = positions_world[:, 0, 1] # We use a xzy coordinate system
    speeds_abs = np.linalg.norm(xy, axis=1) # Instantaneous speed along the trajectory
    amplitude = scipy.ndimage.filters.gaussian_filter1d(speeds_abs, 5) # Low-pass filter
    speeds_abs -= amplitude # Extract high-frequency details
    # Integrate the high-frequency speed component to recover an offset w.r.t. the trajectory
    speeds_abs = np.cumsum(speeds_abs)
    
    xy /= np.linalg.norm(xy, axis=1).reshape(-1, 1) + 1e-9 # Epsilon to avoid division by zero

    return np.stack((speeds_abs, z, # Translations
                     np.cos(phase)*amplitude, np.sin(phase)*amplitude, # Controls
                     xy[:, 0]*amplitude, xy[:, 1]*amplitude, # Controls
                     np.sin(xy_orientation), np.cos(xy_orientation)), # Controls
                    axis=1)

def build_extra_features(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            xy_orientation = dataset[subject][action]['rotations_euler'][:, 0, 1]
            dataset[subject][action]['extra_features'] = extract_translations_controls(
                                                                dataset[subject][action]['positions_world'],
                                                                xy_orientation)
            
def angle_difference_batch(y, x):
    """
    Compute the signed angle difference y - x,
    where x and y are given as versors.
    """
    return np.arctan2(y[:, :, 1]*x[:, :, 0] - y[:, :, 0]*x[:, :, 1], y[:, :, 0]*x[:, :, 0] + y[:, :, 1]*x[:, :, 1])

def phase_to_features(phase_signal):
    """
    Given a [A(t)*cos(phase), A(t)*sin(phase)] signal, extract a set of features:
    A(t), absolute phase (not modulo 2*pi), angular velocity.
    This function expects a (batch_size, seq_len, 2) tensor.
    """
    assert len(phase_signal.shape) == 3
    assert phase_signal.shape[-1] == 2
    amplitudes = np.linalg.norm(phase_signal, axis=2).reshape(phase_signal.shape[0], -1, 1)
    phase_signal = phase_signal/(amplitudes + 1e-9)
    phase_signal_diff = angle_difference_batch(phase_signal[:, 1:], phase_signal[:, :-1])
    frequencies = np.pad(phase_signal_diff, ((0, 0), (0, 1)), 'edge').reshape(phase_signal_diff.shape[0], -1, 1)
    return amplitudes, np.cumsum(phase_signal_diff, axis=1), frequencies

def compute_splines(dataset):
    """
    For each animation in the dataset, this method computes its equal-segment-length spline,
    along with its interpolated tracks (e.g. local speed, footstep frequency).
    """
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            xy = dataset[subject][action]['positions_world'][:, 0, [0, 2]]
            spline = Spline(xy, closed=False)

            # Add extra tracks (facing direction, phase/amplitude)
            xy_orientation = dataset[subject][action]['rotations_euler'][:, 0, 1]
            y_rot = np.array([np.sin(xy_orientation), np.cos(xy_orientation)]).T
            spline.add_track('direction', y_rot, interp_mode='circular')
            phase = dataset[subject][action]['extra_features'][:, [2, 3]]
            amplitude, _, frequency = phase_to_features(np.expand_dims(phase, 0))

            spline.add_track('amplitude', amplitude[0], interp_mode='linear')
            spline.add_track('frequency', frequency[0], interp_mode='linear')

            spline = spline.reparameterize(5, smoothing_factor=1)
            avg_speed_track = spline.get_track('amplitude')
            avg_speed_track[:] = np.mean(avg_speed_track)
            spline.add_track('average_speed', avg_speed_track, interp_mode='linear')
            
            dataset[subject][action]['spline'] = spline