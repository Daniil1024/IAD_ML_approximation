import random
import subprocess
import numpy as np
import os
from multiprocessing import Pool
# run this file when you want to generate a synthetic dataset of values from IAD calculation in a folder

np.random.seed(42)
random.seed(42)

def run_iad(sample_file):
    RAW_PATH = "../Data/DIS_raw"
    PROCESSED_PATH = "../Data/DIS_processed"
    IAD_PATH = "/home/zhitov01/Downloads/iad-latest/iad-3-11-1"
    SAMPLE = sample_file
    print(sample_file)

    SAVE_PATH = PROCESSED_PATH + SAMPLE

    if os.path.exists(SAVE_PATH):
        if os.path.isfile(SAVE_PATH + "/1.rxt") and os.path.isfile(SAVE_PATH + "/1.txt"):
            print(f"\t [INFO] Already computed SAVE_PATH:  {SAVE_PATH}")
            return

    SAVE_FILE_NAME = "1.rxt"

    STANDARD_REFLECTANCE = 0.985
    REFRACTIVE_INDEX_SAMPLE = 1.43
    REFRACTIVE_INDEX_GLASS = 0.0
    THICKNESS_GLASS = 0.0
    BEAM_DIAMETER = 8.0
    WALL_REFLECTANCE = 0.985
    DETECTOR_REFLECTANCE = 0.0
    SPHERE_DIAMETER = 50.0
    SAMPLE_PORT_DIAMETER = 9.9
    ENTRANCE_PORT_DIAMETER = 4.9
    DETECTOR_PORT_DIAMETER = 0.2
    SPHERE_DIAMETER_2 = 50.0
    SAMPLE_PORT_DIAMETER_2 = 9.9
    ENTRANCE_PORT_DIAMETER_2 = 0.0
    DETECTOR_PORT_DIAMETER_2 = 0.2

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    seed = int(sample_file.split(".")[1])
    print("seed: ", str(sample_file))
    np.random.seed(seed)
    random.seed(seed)
    THICKNESS_SAMPLE = 2 + 2 * np.random.random()
    print("Calculated thickness from file: ", THICKNESS_SAMPLE, "mm")

    R_wl = np.ones(500)*600.0
    border = 1e-6
    total_reflectance = border + (1 - 2*border) * np.random.rand(R_wl.shape[0])
    total_transmission = border + (1 - 2*border) * np.random.rand(R_wl.shape[0])
    more_than_one = np.where(total_reflectance + total_transmission >= 1) # about a half of the points will be in the upper triangle
    total_reflectance[more_than_one] = 1 - total_reflectance[more_than_one]
    total_transmission[more_than_one] = 1 - total_transmission[more_than_one]  # imagine a square in R,T coordinates.
    # If a point is in the upper right triangle, we reflect its coordinates w.r.t (0.5,0.5)

    with open(SAVE_PATH + SAVE_FILE_NAME, "w") as rxt_file:
        rxt_file.writelines(f"IAD1 {REFRACTIVE_INDEX_SAMPLE:.6f}"
                            f"\t{REFRACTIVE_INDEX_GLASS:.6f}"
                            f"\t{THICKNESS_SAMPLE:.6f}"
                            f"\t{THICKNESS_GLASS:.6f}"
                            f"\t{BEAM_DIAMETER:.6f}"
                            f"\t{STANDARD_REFLECTANCE:.6f}"
                            f"\t2.000000"
                            f"\t{SPHERE_DIAMETER:.6f}"
                            f"\t{SAMPLE_PORT_DIAMETER:.6f}"
                            f"\t{ENTRANCE_PORT_DIAMETER:.6f}"
                            f"\t{DETECTOR_PORT_DIAMETER:.6f}"
                            f"\t{WALL_REFLECTANCE:.6f}"
                            f"\t{SPHERE_DIAMETER_2:.6f}"
                            f"\t{SAMPLE_PORT_DIAMETER_2:.6f}"
                            f"\t{ENTRANCE_PORT_DIAMETER_2:.6f}"
                            f"\t{DETECTOR_PORT_DIAMETER_2:.6f}"
                            f"\t{WALL_REFLECTANCE:.6f}"
                            f"\t2.000000\n")
        for wl_idx, wl in enumerate(R_wl):
            rxt_file.writelines(f"{wl:.6f} {total_reflectance[wl_idx]:.6f} {total_transmission[wl_idx]:.6f}\n")

    print("Starting IAD program")
    cmd = list()
    cmd.append(IAD_PATH + "/iad")
    cmd.append("-q 12")
    cmd.append("-i 8")
    cmd.append("-g 0.7")
    cmd.append(SAVE_PATH + "1.rxt")
    subprocess.run(cmd)


if __name__ == "__main__":
    samples = []
    for n in range(1, 5000):
        samples.append(f"/high_quality_IAD_data_with_2_spheres_and_thickness/T.{n}.{n}.a/1/")

    with Pool(19) as p:
        p.map(run_iad, samples)