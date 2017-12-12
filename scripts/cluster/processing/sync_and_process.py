#!/usr/bin/env python

import datetime as dt
import functools as ft
import logging
import os
import sys

import simulacra as si
import simulacra.cluster as clu

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_file = f"{__file__.strip('.py')}__{dt.datetime.now().strftime('%Y-%m-%d')}"
cp_logger = si.utils.LogManager(
    '__main__', 'simulacra', 'ionization',
    stdout_logs = True,
    stdout_level = logging.INFO,
    file_logs = False,
    file_level = logging.DEBUG, file_name = log_file, file_dir = os.path.join(os.getcwd(), 'logs'), file_mode = 'a',
)

DROPBOX_PROCESS_NAMES = ['Dropbox.exe']


def synchronize_with_cluster(cluster_interface):
    with si.utils.SuspendProcesses(*DROPBOX_PROCESS_NAMES):
        with cluster_interface as ci:
            js = ci.get_job_status()
            logger.info(js)
            with open('report_cluster.txt', mode = 'w', encoding = 'utf-8') as f:
                f.write(js)

            ci.mirror_remote_home_dir()


def process_job(job_name, jobs_dir = None):
    if jobs_dir is None:
        jobs_dir = os.getcwd()
    job_dir = os.path.join(jobs_dir, job_name)

    job_info = clu.load_job_info_from_file(job_dir)

    try:
        jp = clu.JobProcessor.load(os.path.join(job_dir, f'{job_name}.job'))

        logger.debug('Loaded existing job processor for job {}'.format(job_name))
    except FileNotFoundError:
        jp_type = job_info['job_processor_type']
        jp = jp_type(job_name, job_dir)

        logger.debug(f'Created new job processor of type {jp_type} for job {job_name}')

    if len(jp.unprocessed_sim_names) > 0:
        with si.utils.SuspendProcesses(*DROPBOX_PROCESS_NAMES):
            jp.load_sims(force_reprocess = False)

    jp.save(target_dir = os.path.join(os.getcwd(), 'job_processors'))

    try:
        if len(jp.unprocessed_sim_names) < jp.sim_count:
            jp.summarize()
    except Exception as e:
        logger.exception(e)

    return jp


def process_jobs(jobs_dir):
    job_processors = []

    job_names = [f for f in os.listdir(jobs_dir) if os.path.isdir(os.path.join(jobs_dir, f))]
    logger.debug(f'Found jobs: {", ".join(job_names)}')

    job_processors = [process_job(job_name, jobs_dir) for job_name in job_names]

    total_sim_count = sum(jp.sim_count - len(jp.unprocessed_sim_names) for jp in job_processors)
    total_runtime = sum((jp.running_time for jp in job_processors), dt.timedelta())

    logger.info(f'Processed {len(job_processors)} jobs containing {total_sim_count} simulations, with total runtime {total_runtime}')

    report = generate_processing_report(job_processors)
    print(report)
    with open('report_processing.txt', mode = 'w', encoding = 'utf-8') as f:
        f.write(report)


def generate_processing_report(job_processors):
    len_of_longest_jp_name = max(len(jp.name) for jp in job_processors) if job_processors else 10

    header = f' {"Job Name".center(len_of_longest_jp_name)} │ Finished │ Total │ Runtime'

    bar = ''.join('─' if char != '│' else '┼' for char in header)

    lines_in_progress = []
    lines_finished = []
    for jp in job_processors:
        s = f' {jp.name.ljust(len_of_longest_jp_name)} │ {str(jp.sim_count - len(jp.unprocessed_sim_names)).center(8)} │ {str(jp.sim_count).center(5)} │ {jp.running_time}'

        if len(jp.unprocessed_sim_names) > 0:
            lines_in_progress.append(s)
        else:
            lines_finished.append(s)

    total_processed = sum(jp.sim_count - len(jp.unprocessed_sim_names) for jp in job_processors)
    total_jobs = sum(jp.sim_count for jp in job_processors)
    total_runtime = sum((jp.running_time for jp in job_processors), dt.timedelta())
    footer = f' {" " * len_of_longest_jp_name} │ {str(total_processed).center(8)} │ {str(total_jobs).center(5)} │ {total_runtime}'

    report_components = [
        '',
        header,
        bar,
        *lines_in_progress,
        bar,
        *lines_finished,
        bar,
        footer,
        bar.replace('┼', '┴'),
        ''
    ]
    report = '\n'.join(report_components)

    return report


if __name__ == '__main__':
    with cp_logger as logger:
        try:
            ci = clu.ClusterInterface('submit-5.chtc.wisc.edu', username = 'karpel', key_path = 'E:\chtc_ssh_private')
            jobs_dir = "E:\Dropbox\Research\Cluster\cluster_mirror\home\karpel\jobs"

            si.utils.try_loop(
                ft.partial(synchronize_with_cluster, ci),
                ft.partial(process_jobs, jobs_dir),
                wait_after_success = dt.timedelta(hours = 1),
                wait_after_failure = dt.timedelta(hours = 1),
            )
        except KeyboardInterrupt:
            logger.info('Detected keyboard interrupt, exiting')
            for process_name in DROPBOX_PROCESS_NAMES:
                si.utils.resume_processes_by_name(process_name)
            sys.exit(0)
        except Exception as e:
            logger.exception('Encountered unhandled exception during loop')
            raise e
