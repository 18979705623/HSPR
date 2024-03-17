
import heapq
import json


def construct_rej_obj_prox():
    room_objects = {}
    for i in range(31):
        room_objects[i] = [0] * 1600

    with open('../datasets/labels/house_pano_info.json') as info_file:
        scans = json.load(info_file)
    viewpoint_regions_dict = {}
    for scan in scans:
        for viewpoint, reg_type in scans[scan].items():
            viewpoint_regions_dict[viewpoint] = reg_type

    viewpoint_objects_dict = {}
    with open('../datasets/labels/all_label_with_name.json') as info_file:
        objects_info = json.load(info_file)
    for viewpont in objects_info:
        id = viewpont.split('_')[1]
        if id not in viewpoint_objects_dict:
            viewpoint_objects_dict[id] = set()
        for obj in objects_info[viewpont]['labels'][1:]:
            viewpoint_objects_dict[id].add(obj)
    # Count the number of occurrences of objects in the area
    for viewpoint_id, region_id in viewpoint_regions_dict.items():
        objects = viewpoint_objects_dict.get(viewpoint_id, set())
        for object_id in objects:
            room_objects[region_id][object_id] += 1

    # Count the five objects that appear most frequently in each room type
    top_five_objects = {}
    for room_id, object_counts in room_objects.items():
        top_five = heapq.nlargest(5, range(len(object_counts)), key=object_counts.__getitem__)
        top_five_objects[room_id] = top_five

    with open("../datasets/reg_obj_proximity.json", "w") as f:
        json.dump(top_five_objects, f)
