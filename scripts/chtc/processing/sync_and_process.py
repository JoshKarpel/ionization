#!/usr/bin/env python3

import datetime
import functools
import logging
import os
import sys

import simulacra as si
import simulacra.cluster as clu

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

LOG_FILE = f"{__file__.strip('.py')}__{datetime.datetime.now().strftime('%Y-%m-%d')}"
LOGMAN = si.utils.LogManager(
    "__main__",
    "simulacra",
    "ionization",
    stdout_logs=True,
    stdout_level=logging.INFO,
    file_logs=False,
    file_level=logging.DEBUG,
    file_name=LOG_FILE,
    file_dir=os.path.join(os.getcwd(), "logs"),
    file_mode="a",
)

DROPBOX_PROCESS_NAMES = ["Dropbox.exe"]


def synchronize_with_cluster(cluster_interface):
    with si.utils.SuspendProcesses(*DROPBOX_PROCESS_NAMES):
        with cluster_interface as ci:
            js = get_job_status(cluster_interface)
            logger.info(js)
            with open("report_cluster.txt", mode="w", encoding="utf-8") as f:
                f.write(js)

            ci.mirror_dir()


def get_job_status(cluster_interface):
    """Get the status of jobs on the cluster."""
    cmd_output = cluster_interface.cmd("condor_q -wide")

    status = cmd_output.stdout.readlines()

    status_str = "Job Status:\n" + "".join(status[1:])

    return status_str


def process_jobs(jobs_dir):
    job_names = [
        f for f in os.listdir(jobs_dir) if os.path.isdir(os.path.join(jobs_dir, f))
    ]
    logger.debug(f'Found jobs: {", ".join(job_names)}')

    job_processors = [
        si.utils.run_in_process(process_job, args=(job_name, jobs_dir))
        for job_name in job_names
    ]  # avoid memory leaks

    total_sim_count = sum(
        jp.sim_count - len(jp.unprocessed_sim_names) for jp in job_processors
    )
    total_runtime = sum(
        (jp.running_time for jp in job_processors), datetime.timedelta()
    )

    logger.info(
        f"Processed {len(job_processors)} jobs containing {total_sim_count} simulations, with total runtime {total_runtime}"
    )

    report = make_processing_report(job_processors)
    print(report)
    with open("report_processing.txt", mode="w", encoding="utf-8") as f:
        f.write(report)


def process_job(job_name, jobs_dir=None):
    with LOGMAN as logger:
        if jobs_dir is None:
            jobs_dir = os.getcwd()
        job_dir = os.path.join(jobs_dir, job_name)

        job_info = clu.load_job_info_from_file(job_dir)

        try:
            jp = clu.JobProcessor.load(os.path.join(job_dir, f"{job_name}.job"))

            logger.debug("Loaded existing job processor for job {}".format(job_name))
        except FileNotFoundError:
            jp_type = job_info["job_processor_type"]
            jp = jp_type(job_name, job_dir)

            logger.debug(
                f"Created new job processor of type {jp_type} for job {job_name}"
            )

        if len(jp.unprocessed_sim_names) > 0:
            with si.utils.SuspendProcesses(*DROPBOX_PROCESS_NAMES):
                jp.load_sims(force_reprocess=False)

        jp.save(target_dir=os.path.join(os.getcwd(), "job_processors"))

        try:
            if len(jp.unprocessed_sim_names) < jp.sim_count:
                jp.summarize()
                jp.make_velocity_plot()
        except Exception as e:
            logger.exception(e)

        return jp


def make_processing_report(job_processors):
    total_processed = sum(
        jp.sim_count - len(jp.unprocessed_sim_names) for jp in job_processors
    )
    total_jobs = sum(jp.sim_count for jp in job_processors)
    total_runtime = sum(
        (jp.running_time for jp in job_processors), datetime.timedelta()
    )

    report = si.utils.table(
        ["Job", "Processed", "Total", "Runtime"],
        [
            *[
                (
                    jp.name,
                    jp.sim_count - len(jp.unprocessed_sim_names),
                    jp.sim_count,
                    jp.running_time,
                )
                for jp in job_processors
            ],
            None,
            ["", total_processed, total_jobs, total_runtime],
        ],
    )

    return report


if __name__ == "__main__":
    with LOGMAN as logger:
        try:
            ci = clu.ClusterInterface(
                "submit-5.chtc.wisc.edu",
                username="karpel",
                key_path="E:\chtc_ssh_private",
                local_mirror_root="mirror",
            )
            jobs_dir = "E:\Dropbox\Research\Cluster\mirror\home\karpel\jobs"

            si.utils.try_loop(
                functools.partial(synchronize_with_cluster, ci),
                functools.partial(process_jobs, jobs_dir),
                wait_after_success=datetime.timedelta(hours=3),
                wait_after_failure=datetime.timedelta(minutes=10),
            )
        except KeyboardInterrupt:
            logger.critical("Detected keyboard interrupt. Exiting...")
            for process_name in DROPBOX_PROCESS_NAMES:
                si.utils.resume_processes_by_name(process_name)
            sys.exit(0)
        except Exception as e:
            logger.exception("Encountered unhandled exception during loop")
            raise e
