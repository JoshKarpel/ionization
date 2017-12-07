import datetime as dt
import functools as ft
import logging
import os

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
    with cp_logger as logger:
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
            jp.summarize()
        except Exception as e:
            logger.exception(e)

        return jp


def process_jobs(jobs_dir):
    job_processors = []

    for job_name in (f for f in os.listdir(jobs_dir) if os.path.isdir(os.path.join(jobs_dir, f))):
        try:
            logger.debug('Found job {}'.format(job_name))
            jp = si.utils.run_in_process(process_job, args = (job_name, jobs_dir))
            job_processors.append(jp)
        except Exception as e:
            logger.exception('Encountered exception while processing job {}'.format(job_name))

    total_sim_count = sum(jp.sim_count - len(jp.unprocessed_sim_names) for jp in job_processors)
    total_runtime = sum((jp.running_time for jp in job_processors), dt.timedelta())

    logger.info(f'Processed {len(job_processors)} jobs containing {total_sim_count} simulations, with total runtime {total_runtime}')

    longest_jp_name_len = max(len(jp.name) for jp in job_processors) if job_processors else 10

    header = f' {"Job Name".center(longest_jp_name_len)} │ Finished │ Total │ Runtime'
    bar = ''.join('─' if char != '│' else '┼' for char in header)
    lines = []
    for jp in job_processors:
        lines.append(f' {jp.name.ljust(longest_jp_name_len)} │ {str(jp.sim_count - len(jp.unprocessed_sim_names) ).center(8)} │ {str(jp.sim_count).center(5)} │ {jp.running_time}')

    report = '\n'.join((header, bar, *lines))

    print(report)

    with open('report_processing.txt', mode = 'w', encoding = 'utf-8') as f:
        f.write(report)


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
        except Exception as e:
            logger.exception('Encountered unhandled exception during loop')
            raise e
