# coding:utf-8

'''
requirements: sox (http://sox.sourceforge.net)
'''

import subprocess
import os
import argparse


def main_loop(data_paths, save_dir, channel_ix):
    for file_path in data_paths:
        print(file_path)
        file_name_wo_ext, ext = os.path.splitext(os.path.basename(file_path))
        subprocess.check_call(["sox", file_path, os.path.join(save_dir, file_name_wo_ext+"_{0:02d}_ch".format(channel_ix)+ext), "remix", "1"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_paths', type=str, help='Path to wav file(s). Regular expression accepted.', nargs='+')
    parser.add_argument('-s','--save_dir', type=str, default='./', help= 'Path to the directory to save mono channel file(s). Current directory by default.')
    parser.add_argument('-c', '--channel_ix', type=str, default="1", help='ID # of channel to extract. 1st by default')
    args = parser.parse_args()

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    main_loop(args.data_paths, args.save_dir, args.channel_ix)