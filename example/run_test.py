import os, os.path
import numpy as np
from tqdm import tqdm

PYTHON = 'python3'
SCRIPT = '../nn-train/test.py'
CFG_PATH = 'nn-train/test'
LOG_PATH = 'nn-train/test'

def run(task_index, molecule):
    os.system('CUDA_VISIBLE_DEVICES=0 %s %s %s/%s/%s.cfg' % (
        PYTHON, SCRIPT, 
        CFG_PATH, molecule, task_index))


def main():
    print('====Task starts====')

    tasks_range = range(500, 901, 40)[0:1]
    tasks = np.array(['d%04d' % (i) for i in tasks_range])

    with tqdm(total=len(tasks)) as pbar:
        for i, task in enumerate(tasks):
            os.system('mkdir -pv %s/H2/%s' % (LOG_PATH, task))
            pbar.set_description('H2/%s' % (task))
            run(task, 'H2')
            pbar.update(1)

    with tqdm(total=len(tasks)) as pbar:
        for i, task in enumerate(tasks):
            os.system('mkdir -pv %s/HeH+/%s' % (LOG_PATH, task))
            pbar.set_description('HeH+/%s' % (task))
            run(task, 'HeH+')
            pbar.update(1)

    print('\n====Task completed====')

if __name__ == '__main__':
    main()
