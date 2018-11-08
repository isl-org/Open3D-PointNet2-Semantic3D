# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--file', help="file xzy rgb r'g'b' label ")
parser.add_argument('--out', help="output directory")
FLAGS = parser.parse_args()

my_data = np.genfromtxt(FLAGS.file, delimiter=' ')
filename = FLAGS.file[0:-8]
print(my_data.shape)
np.savez(
    os.path.join(FLAGS.out,
                 os.path.basename(filename) + "_vertices").encode('utf_8'),
    my_data[:, 0:3])
np.savez(
    os.path.join(FLAGS.out,
                 os.path.basename(filename) + "_colors").encode('utf_8'),
    my_data[:, 3:6].astype(int))
if my_data.shape[1] > 9:
    np.savez(
        os.path.join(FLAGS.out,
                     os.path.basename(filename) + "_labels").encode('utf_8'),
        my_data[:, 9].astype(int))
