
import itertools
import json
import numpy as np
from .calcu_prox import calculated_obj_proximity
np.set_printoptions(threshold=np.inf)


def object_counting(obj_lists):
    count_matric = np.zeros((1600, 1600))
    for lst in obj_lists:
        for comb in itertools.combinations(lst[1:], 2):
            x, y = comb
            count_matric[x, y] += 1
            count_matric[y, x] += 1
    return count_matric


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def construct_obj_prox():
    with open('../datasets/labels/all_label_with_name.json') as info_file:
        pano = json.load(info_file)
    object_list = set()
    for sub_view in pano:
        object_list.add(tuple(pano[sub_view]['labels']))
    object_list = [list(lst) for lst in object_list]
    obj_count_matrix = object_counting(object_list).astype(int)
    obj_pro = calculated_obj_proximity(obj_count_matrix)
    # print(obj_count_matrix[:][0], obj_count_matrix.shape)

    np.save("../datasets/obj_proximity.npy", obj_pro)
    # pro = np.load("../datasets/obj_proximity.npy")
    # print(check_symmetric(pro))
