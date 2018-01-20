import logging
import os

import simulacra as si
import simulacra.cluster as clu
import simulacra.units as u

import ionization as ion
import ionization.cluster as iclu

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_DIR = os.path.join(os.getcwd(), 'out', FILE_NAME)

if __name__ == '__main__':
    with si.utils.LogManager('simulacra', 'ionization', stdout_logs = True, stdout_level = logging.DEBUG) as logger:
        # jp = clu.JobProcessor.load('paper_tdse_test_small_mesh.job')
        jp = clu.JobProcessor.load('tdse_gaussian_pw_scan.job')
        # jp = clu.JobProcessor.load('tdse_heatmap_pw_vs_amp__lin.job')
        # jp = clu.JobProcessor.load('tdse_heatmap_pw_vs_flu__lin.job')  # doesn't generate log y plots because we have 0 fluence in the dataset - good to realize!

        print(jp)
        jp.job_dir_path = OUT_DIR

        jp.summarize()
