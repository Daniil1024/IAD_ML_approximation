import random
import subprocess
from os.path import exists

import numpy as np
import os
from os import path
from multiprocessing import Pool

import PATH

np.random.seed(42)
random.seed(42)
def run_mcml(sample_path):
    SAMPLE = sample_path
    print(sample_path)

    SAVE_PATH = PATH.PROCESSED_PATH + SAMPLE

    SAVE_FILE_NAME = "input.mci"

    REFRACTIVE_INDEX_SAMPLE = 1.43
    ASSUMED_ANISOTROPY = 0.7
    #  BEAM_DIAMETER = 8.0

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    seed = int(sample_path.split(".")[1])
    print("seed: ", str(sample_path))
    np.random.seed(seed)
    random.seed(seed)
    THICKNESS_SAMPLE = 3.0  # 2 + 2 * np.random.random()
    print("Calculated thickness from file: ", THICKNESS_SAMPLE, "mm")
    N = 100
    log_mua = -2.5 + (2 - -2.5) * np.random.rand(N)  # randomised log_mua
    log_mus = 0 + (6 - 0) * np.random.rand(N)  # randomised log_mus
    mua = np.exp(log_mua)
    mus = np.exp(log_mus)
    if not exists(SAVE_PATH+SAVE_FILE_NAME):
        f = open(SAVE_PATH+SAVE_FILE_NAME, 'w')
        f.close()
    with open(SAVE_PATH + SAVE_FILE_NAME, "rb+") as mci_file:
        mci_file.write(b"1.0 #file version\n")
        mci_file.write((str(N)+"\t #number of runs\n\n").encode('ascii'))
        for i in range(N):
            mci_file.write((SAVE_PATH+str(i)+'.mco A\n').encode('ascii'))
            mci_file.write(b"10000\n") # does not accept scientific format
            mci_file.write(b"0.01 0.01\n")
            mci_file.write(b"40 50 1\n\n")
            mci_file.write(b"1\n")  # number of layers
            mci_file.write(b"1\n")  # refractive index of the above layer
            mci_file.write((str(REFRACTIVE_INDEX_SAMPLE) + " ").encode('ascii'))
            mci_file.write(f"{mua[i]:8f} ".encode('ascii'))
            mci_file.write(f"{mus[i]:8f} ".encode('ascii'))
            mci_file.write(f"{ASSUMED_ANISOTROPY:8f} ".encode('ascii'))
            mci_file.write(f"{THICKNESS_SAMPLE/10:8f}\n".encode('ascii'))
            mci_file.write(b"1\n\n")

    cmd = list()
    cmd.append(PATH.MCML_PATH)
    cmd.append(SAVE_PATH + SAVE_FILE_NAME)
    print("Starting MCML program\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    os.chdir("../")
    samples = []
    for n in range(5):
        samples.append(f"/new_MCML/T.{n}.{n}.a/1/")

    with Pool(19) as p:
        p.map(run_mcml, samples)
