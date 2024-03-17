
import math

import numpy as np
import torch


def single_non_zero_to_one(lst):
    indices = [i for i, x in enumerate(lst) if x != 0]
    if len(indices) == 1:
        # print(lst)
        return [1 if i == indices[0] else 0 for i in range(len(lst))]
    else:
        return lst


def activate_function(input, mid, max, top):
    output = []
    for x in input:
        if x == 0:
            output.append(0)
        elif x < mid:
            y = - pow(x, 2) / (2 * pow(mid, 2)) + x / mid
            output.append(y)
        elif mid <= x <= max:
            y = pow(x, 2) / (2 * pow((mid - top), 2)) - (mid * x) / pow((mid - top), 2) + pow(mid, 2) / (
                    2 * pow((mid - top), 2)) + 0.5
            output.append(y)
        else:
            y = float(torch.arctan(torch.tensor(x, dtype=torch.float)) - (math.pi / 2 - 0.92))
            output.append(y)
    output = single_non_zero_to_one(output)
    return output


def calculated_reg_proximity(c_matrix):
    # print(c_matrix)
    p_matrix = []
    for i, count_list in enumerate(c_matrix):
        all_list = torch.clone(count_list)
        count_list = count_list.masked_select(count_list > 0)
        while len(count_list) <= 4:
            count_list = torch.cat([count_list, torch.tensor([0])], 0)  # fill
        count_list = count_list.tolist()
        j = 0
        while j < 2:
            count_list.remove(max(count_list))
            count_list.remove(min(count_list))
            j += 1

        mid_count = np.mean(count_list)
        max_count = max(count_list)
        top_count = max_count + mid_count
        prob_list = activate_function(all_list.tolist(), mid_count, max_count, top_count)
        p_matrix.append(prob_list)
    conn_pro = np.around(np.array(p_matrix), 4) + np.eye(31, dtype=np.float32)
    conn_pro = conn_pro.T
    # conn_pro_flip = np.flip(conn_pro, axis=(0, 1))
    # for i, row in enumerate(conn_pro_flip):
    #     conn_pro_flip[:, i] = row
    # conn_pro = np.flip(conn_pro_flip, axis=(0, 1))
    return conn_pro


def calculated_obj_proximity(o_matrix):
    p_matrix = []
    for i, count_list in enumerate(o_matrix):
        all_list = np.copy(count_list)
        count_list = count_list[count_list > 0]
        while len(count_list) <= 20:
            count_list = np.append(count_list, 0)
        count_list = count_list.tolist()
        j = 0
        while j < 10:
            count_list.remove(max(count_list))
            count_list.remove(min(count_list))
            j += 1
        mid_count = np.mean(count_list)
        max_count = max(count_list)
        top_count = max_count + mid_count
        prob_list = activate_function(all_list.tolist(), mid_count, max_count, top_count)
        p_matrix.append(prob_list)
    conn_pro = np.around(np.array(p_matrix), 4) + np.eye(1600, dtype=np.float32)
    return conn_pro
