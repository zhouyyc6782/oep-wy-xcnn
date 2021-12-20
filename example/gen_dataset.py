from __future__ import print_function
import numpy as np
import os
import sys


PATH_PREFIX = "./oep-wy/dataset"

raw_files_H2 = ["%s/H2/%s" % (PATH_PREFIX, f) for f in os.listdir("%s/H2" % (PATH_PREFIX)) if (not f.endswith("coords.npy")) and f.endswith(".npy")]
raw_files_H2 = sorted(raw_files_H2)
raw_files = raw_files_H2
#raw_files_HeH = ["%s/HeH+/%s" % (PATH_PREFIX, f) for f in os.listdir("%s/HeH+" % (PATH_PREFIX)) if (not f.endswith("coords.npy")) and f.endswith(".npy")]
#raw_files_HeH = sorted(raw_files_HeH)
#raw_files = raw_files_H2 + raw_files_HeH

all_data = np.load(raw_files[0])
for i, f in enumerate(raw_files[1:]):
    new_data = np.load(f)
    all_data = np.concatenate((all_data, new_data), axis=0)
assert(all_data.shape[1] == 4 * 9 * 9 * 9 + 1)

print("Dataset size:", all_data.shape)

np.random.shuffle(all_data)
np.save("%s/H2_0.9_9" % (PATH_PREFIX), all_data)

