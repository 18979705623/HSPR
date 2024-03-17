import re
import json

import torch


def save(dict):
    if isinstance(dict, str):
        dict = eval(dict)
    with open('../datasets/labels/sub_region_label.txt', 'w', encoding='utf-8') as f:
        # f.write(str(dict))
        str_ = json.dumps(dict, ensure_ascii=False)
        print(type(str_), str_)
        f.write(str_)


with open('../datasets/labels/candidate_label.json') as info_file:
    candidate_list = json.load(info_file)


def area_segmentation(area_list, room_type):
    area_dict = {}
    for i, area in enumerate(area_list):
        area_dict[str(room_type) + '_' + str(i)] = area
    return area_dict


def get_other_area(current_area_key, all_area):
    room_type = current_area_key.split('_')[0]
    for i, area in enumerate(list(all_area)):
        if re.match(room_type + '_', area):
            all_area.pop(area)
    return all_area


def room_subarea(scan, viewpoint_lists):
    '''Subregion segmentation: All nodes with the same region type are classified in detail.'''
    area_dict = {}
    sub_area_dict = {}
    for i, (viewpoint, room_type) in enumerate(viewpoint_lists.items()):
        if room_type not in area_dict.keys():
            area_dict[room_type] = []
            area_dict[room_type].append(viewpoint)
        else:
            area_dict[room_type].append(viewpoint)

    for i, (room_type, vp_list) in enumerate(area_dict.items()):
        tmp = []  # Stores the disconnected viewpoints in the same area of each scan
        num_set = 0  # The number of sets in tmp
        for vp in vp_list:
            try:
                cands = candidate_list[scan][vp]
            except:
                continue
            # The intersection of such an viewpoint with an adjacent node of the preceding viewpoint
            common_viewpoints = set(vp_list).intersection(set(cands))
            add_count, last_index, null_set_count = 0, 0, 0
            #  The number of times a common node is added to each set in tmp; The index of the set where the most recent add operation was performed;
            #  Number of times that the common viewpoint and tmp_set are detected as empty sets.
            for tmp_set in tmp:
                if common_viewpoints.isdisjoint(tmp_set):
                    null_set_count += 1
                else:
                    break
            # If the intersection with all set sets in tmp is empty, a new set is created and the current node vp and common node are added to it.
            if null_set_count == len(tmp):
                tmp.append(set())
                num_set += 1
                tmp[num_set - 1].add(vp)
                [tmp[num_set - 1].add(vpt) for vpt in common_viewpoints]
            else:  # Otherwise it is added to the set with the intersection.
                for j, tmp_set in enumerate(tmp):
                    if not common_viewpoints.isdisjoint(tmp_set):
                        add_count += 1
                        if add_count > 1:  # If it is added more than twice, it means that all sets in tmp are connected. And put them back together.
                            tmp[last_index].update(tmp_set)
                            del tmp[j]
                            num_set -= 1
                        else:
                            tmp[j].add(vp)
                            [tmp[j].add(vpt) for vpt in common_viewpoints]
                            last_index = j
        sub_area_dict.update(area_segmentation(tmp, room_type))
    return sub_area_dict


def frequency_statistics(scanId, sub_area_dict):
    count_matrix = torch.zeros(31, 31)
    for sub_area_key, sub_area_value in sub_area_dict.items():
        connected_relation = set()
        for vpt in list(sub_area_value):
            cand = candidate_list[scanId][vpt]
            all_area_dict = sub_area_dict.copy()
            other_area_dict = get_other_area(sub_area_key, all_area_dict)
            for other_area_key, other_area_value in other_area_dict.items():
                if not set(cand).isdisjoint(other_area_value):
                    connected = sub_area_key.split('_')[0] + '_' + other_area_key
                    connected_relation.add(connected)
        for i, connected in enumerate(connected_relation):
            room_type0 = int(connected.split('_')[0])
            room_type1 = int(connected.split('_')[1])
            count_matrix[room_type0][room_type1] += 1
    return count_matrix
