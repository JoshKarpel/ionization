import os
import subprocess

import simulacra.cluster as clu


def generate_chtc_submit_string(
    job_name: str, specification_count: int, do_checkpoints: bool = True
) -> str:
    """
    Return a formatted submit string for an HTCondor job.

    Parameters
    ----------
    job_name : :class:`str`
        The name of the job.
    specification_count : :class:`int`
        The number of specifications in the job.
    do_checkpoints : :class:`bool`
        If the simulations are going to use checkpoints, this should be ``True``.

    Returns
    -------
    str
        An HTCondor submit string.
    """
    with open("submit_template.sub") as f:
        submit_template = f.read()

    format_data = dict(
        batch_name=clu.ask_for_input("Job batch name?", default=job_name, cast_to=str),
        checkpoints=str(do_checkpoints).lower(),
        flockglide=str(clu.ask_for_bool("Flock and Glide?", default="y")).lower(),
        memory=clu.ask_for_input("Memory (in GB)?", default=1, cast_to=float),
        disk=clu.ask_for_input("Disk (in GB)?", default=5, cast_to=float),
        max_idle=clu.ask_for_input("Max Idle Jobs?", default=1000, cast_to=int),
        num_jobs=specification_count,
    )

    return submit_template.format(**format_data)


def submit_check(submit_string: str):
    """Ask the user whether the submit string looks correct."""
    print("-" * 20)
    print(submit_string)
    print("-" * 20)

    check = clu.ask_for_bool("Does the submit file look correct?", default="No")
    if not check:
        clu.abort_job_creation()


def write_submit_file(submit_string: str, job_dir: str):
    """Write the submit string to a file."""
    print("Writing submit file...")

    with open(
        os.path.join(job_dir, "submit_job.sub"), mode="w", encoding="utf-8"
    ) as file:
        file.write(submit_string)


def submit_job(job_dir: str, factory: bool = False):
    """Submit a job using a pre-existing submit file."""
    print("Submitting job...")

    os.chdir(job_dir)

    cmds = ["condor_submit", "submit_job.sub"]

    if factory:
        cmds.append("-factory")

    subprocess.run(cmds)

    os.chdir("..")
