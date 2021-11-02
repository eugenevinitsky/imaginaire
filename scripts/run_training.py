#!/usr/bin/env python3
import time
import os
import sys
import argparse
import pathlib, shutil
from datetime import datetime
from subprocess import Popen, DEVNULL


class Overrides(object):
    def __init__(self):
        self.kvs = dict()

    def add(self, key, values):
        value = ','.join(str(v) for v in values)
        assert key not in self.kvs
        self.kvs[key] = value

    def cmd(self):
        cmd = []
        for k, v in self.kvs.items():
            cmd.append(f'{k}={v}')
        return cmd


def make_code_snap(experiment, code_path, slurm_dir='exp'):
    if len(code_path) > 0:
        snap_dir = pathlib.Path(code_path) / slurm_dir
    else:
        snap_dir = pathlib.Path.cwd() / slurm_dir
    # snap_dir /= now.strftime('%Y.%m.%d')
    snap_dir /= f'_{experiment}'
    snap_dir.mkdir(exist_ok=True, parents=True)
    
    def copy_dir(dir, pat):
        dst_dir = snap_dir / 'code' / dir
        dst_dir.mkdir(exist_ok=True, parents=True)
        for f in (src_dir / dir).glob(pat):
            try:
                shutil.copytree(f, dst_dir / f.name)
            except:
                shutil.copy(f, dst_dir / f.name)
    
    dirs_to_copy = ['./configs/projects/munit/mujoco', #'./dataset/mujoco_raw/train',
                    './configs/projects/munit/robonet', #'./dataset/robonet_raw/train',
                    './imaginaire']
    src_dir = pathlib.Path.cwd()
    for dir in dirs_to_copy:
        copy_dir(dir, '*')
    copy_dir('./', 'train_copy.py')
    copy_dir('./', 'main_worker.py')
    copy_dir('./', 'config.yaml')
    
    return snap_dir 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('cfg_path', type=str)
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()
   
    now = datetime.now()
    curr_date = now.strftime('%Y.%m.%d')
    curr_time = now.strftime('%H%M%S')
    save_path = f'/checkpoint/eugenevinitsky/rl_model_gen/generators/sweep/{curr_date}/{curr_time}_{args.experiment}'
    snap_dir = make_code_snap(args.experiment, save_path)
    print(str(snap_dir))
    overrides = Overrides()
    overrides.add('gpus_per_node', [4])
    # TODO(eugenevinitsky) get the cfg path from snag
    overrides.add('config', [args.cfg_path])
    overrides.add('hydra/launcher', ['submitit_slurm'])
    overrides.add('hydra.launcher.partition', ['learnlab'])
    overrides.add('hydra.sweep.dir', [save_path])
    overrides.add('experiment', [args.experiment])
    
    cmd = ['python', str(snap_dir / 'code' / 'train_copy.py'), '-m']
    print(cmd)
    cmd += overrides.cmd()

    if args.dry:
        print(' '.join(cmd))
    else:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(snap_dir / 'code')
        p = Popen(cmd, env=env)
        p.communicate()


if __name__ == '__main__':
    main()