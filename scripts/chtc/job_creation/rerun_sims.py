#!usr/bin/env python3

import os
import argparse
import subprocess

import simulacra as si
import simulacra.cluster as clu


def get_missing_sim_names():
    input_names = set(f.rstrip('.spec') for f in os.listdir('inputs/'))
    output_names = set(f.rstrip('.sim') for f in os.listdir('outputs/'))

    return input_names - output_names


def generate_sim_names_file(sim_name_file_name, sim_names):
    with open(sim_name_file_name, mode = 'w') as sim_name_file:
        sim_name_file.write('\n'.join(sorted(sim_names)))


def generate_rerun_submit_file(submit_file_name, sim_name_file_name):
    # get the existing submit file
    with open('submit_job.sub', mode = 'r') as submit_file:
        submit_lines = submit_file.readlines()

    # last line is the queue command, which we need to replace
    submit_lines[-1] = f'queue simname from {sim_name_file_name}'

    # recombine
    rerun_str = ''.join(submit_lines)

    # replace $(Process) with $(jobnumber)
    rerun_str = rerun_str.replace('$(Process)', '$(simname)')

    with open(submit_file_name, mode = 'w') as rerun_submit_file:
        rerun_submit_file.write(rerun_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Rerun simulations from a job.')
    parser.add_argument('job_name',
                        type = str,
                        help = 'the name of the job')
    parser.add_argument('sim_names',
                        nargs = '*')
    parser.add_argument('--missing', '-m',
                        action = 'store_true')

    args = parser.parse_args()

    os.chdir(args.job_name)

    rerun_identifier = f'rerun_{si.utils.get_now_str()}'
    rerun_submit_file_name = f'submit_{rerun_identifier}.sub'
    sim_name_file_name = f'simnames_for_{rerun_identifier}.txt'

    generate_rerun_submit_file(rerun_submit_file_name, sim_name_file_name)

    sim_names = set(args.sim_names)
    if args.missing:
        sim_names = sim_names.union(get_missing_sim_names())

    print('-' * 50)
    for sim_name in sorted(sim_names):
        print(sim_name)
    print('-' * 50)

    if clu.ask_for_bool(f'Rerunning above sim names (total {len(sim_names)}. Ok?', default = 'n'):
        generate_sim_names_file(sim_name_file_name, sim_names)
        cmds = [
            'condor_submit',
            rerun_submit_file_name,
        ]

        subprocess.run(cmds)
