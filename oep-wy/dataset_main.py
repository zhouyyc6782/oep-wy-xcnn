from __future__ import print_function
import numpy
import os
import sys

from Config import get_options
from wy import WY
from density import *
from potential import *


def dataset():



def main():
    oep_opts = get_options(sys.argv[1], 'OEP')
    dataset_opts = get_options(sys.argv[1], 'DATASET')

    for k, v in oep_opts.items():
        print(k, '\t', v, '\t', v.__class__)
    print()
    for k, v in datset_opts.items():
        print(k, '\t', v, '\t', v.__class__)

    print('==== OEP ====')
    wy = WY(oep_opts)
    wy.OEP()
    print()

    

