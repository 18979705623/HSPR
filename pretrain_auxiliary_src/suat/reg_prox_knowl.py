
import json
import numpy as np
import torch

from .get_candidate import generate_candidate_labels
from .reg_sub_area import room_subarea, frequency_statistics
from .calcu_prox import calculated_reg_proximity


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def construct_reg_prox():
    with open('../datasets/labels/house_pano_info.json') as info_file:
        scans = json.load(info_file)
    generate_candidate_labels()
    count_matrix = torch.zeros(31, 31)
    for scan in scans:
        sub_area_dict = room_subarea(scan, scans[scan])
        count_matrix += frequency_statistics(scan, sub_area_dict)
    conn_pro = calculated_reg_proximity(count_matrix)

    np.save("../datasets/reg_proximity.npy", conn_pro)
    # pro = np.load("../datasets/reg_proximity.npy")
    # print(check_symmetric(pro))
