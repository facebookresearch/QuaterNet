# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import os
import errno
import zipfile
import numpy as np
from urllib.request import urlretrieve
from shutil import rmtree
from common.quaternion import qfix

if __name__ == '__main__':
    output_directory = 'datasets'
    output_filename = 'dataset_locomotion'
    locomotion_dataset_url = 'http://theorangeduck.com/media/uploads/other_stuff/motionsynth_data.zip'

    try:
        # Create output directory if it does not exist
        os.makedirs(output_directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    output_file_path = output_directory + '/' + output_filename
    if os.path.exists(output_file_path + '.npz'):
        print('The dataset already exists at', output_file_path + '.npz')
    else:
        # Download locomotion dataset by Holden et  al.
        print('Downloading dataset by Holden et al. (it may take a while)...')
        locomotion_path = output_directory + '/motionsynth_data.zip'
        urlretrieve(locomotion_dataset_url, locomotion_path)
        print('Extracting locomotion dataset...')
        with zipfile.ZipFile(locomotion_path, 'r') as archive:
            archive.extractall(output_directory)
        os.remove(locomotion_path) # Clean up

        print('Converting format...')
        sys.path.append(output_directory + '/motionsynth_data/motion')
        import BVH as BVH
        import Animation as Animation
        from Quaternions import Quaternions
        from Pivots import Pivots

        def process_file(filename):
            # Find class
            cls = os.path.splitext(os.path.split(filename)[1])[0][11:-8]

            anim, names, frametime = BVH.load(filename)
            global_positions = Animation.positions_global(anim)

            pos = anim.positions[:, 0] # Root joint trajectory
            rot = qfix(anim.rotations.qs) # Local joint rotations as quaternions
            return pos, rot, cls


        def get_files(directory):
            files = []
            for f in sorted(list(os.listdir(directory))):
                name = os.path.join(directory, f)
                if os.path.isfile(name) and f.endswith('.bvh') and f != 'rest.bvh':
                    files.append(name)
            return files

        locomotion_files = get_files(output_directory + '/motionsynth_data/data/processed/edin_locomotion')
        locomotion_pos = []
        locomotion_rot = []
        locomotion_subjects = []
        locomotion_actions = []
        counters = {}
        for i, item in enumerate(locomotion_files):
            pos, rot, cls = process_file(item)
            if cls not in counters:
                counters[cls] = 1
            else:
                counters[cls] += 1
            locomotion_pos.append(pos)
            locomotion_rot.append(rot)
            locomotion_subjects.append('S1') # We have no information about the subjects in this dataset
            locomotion_actions.append(cls + '_' + str(counters[cls]))
        np.savez_compressed(output_file_path,
                            trajectories=locomotion_pos,
                            rotations=locomotion_rot,
                            subjects=locomotion_subjects,
                            actions=locomotion_actions)
        rmtree(output_directory + '/motionsynth_data') # Clean up
        print('Done.')