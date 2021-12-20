from __future__ import print_function
import os, os.path

PYTHON = "python3"
SCRIPT = "../nn-train/main.py"
CFG_PATH = "nn-train/train/train.cfg"


cmd = "ln -sv oep-wy/dataset nn-train/"
os.system(cmd)

cmd = "CUDA_VISIBLE_DEVICES=1 %s %s %s" % (PYTHON, SCRIPT, CFG_PATH)
print(cmd)
os.system(cmd)

