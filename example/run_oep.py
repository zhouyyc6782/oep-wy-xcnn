import os, os.path
import numpy as np
from tqdm import tqdm

PYTHON = 'python'
SCRIPT = '../oep-wy/main.py'
CFG_PATH = 'oep-wy/config'
LOG_PATH = 'oep-wy/log'
# 01;37m light white


def run(task_index, molecule):
    os.system('%s %s %s/%s/%s.cfg 1>%s/%s/%s.log 2>%s/%s/%s.err' % (
        PYTHON, SCRIPT, 
        CFG_PATH, molecule, task_index, 
        LOG_PATH, molecule, task_index, 
        LOG_PATH, molecule, task_index))


def main():
    print('====Task starts====')
    os.system('mkdir -pv %s/H2 %s/HeH+' % (LOG_PATH, LOG_PATH))

    tasks_range = range(500, 901, 40)
    tasks = np.array(['d%04d' % (i) for i in tasks_range])

    with tqdm(total=len(tasks)) as pbar:
        for i, task in enumerate(tasks):
            pbar.set_description('H2/%s' % (task))
            run(task, 'H2')
            pbar.update(1)

    with tqdm(total=len(tasks)) as pbar:
        for i, task in enumerate(tasks):
            pbar.set_description('HeH+/%s' % (task))
            run(task, 'HeH+')
            pbar.update(1)

    print('\n====Task completed====')

if __name__ == '__main__':
    main()
