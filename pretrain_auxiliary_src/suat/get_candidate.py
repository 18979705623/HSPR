
import json
import math
import numpy as np
from utils.data import load_nav_graphs, new_simulator
sim = new_simulator('../datasets/R2R/connectivity')
buffered_state_dict = {}
all_candidate_label = {}
def make_candidate(scanId, viewpointId):
    ''' Obtain information about adjacent nodes.'''
    def _loc_distance(loc):
        return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)

    adj_dict = {}
    long_id = "%s_%s" % (scanId, viewpointId)
    if long_id not in buffered_state_dict:
        for ix in range(36):
            if ix == 0:
                sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
            elif ix % 12 == 0:
                sim.makeAction([0], [1.0], [1.0])
            else:
                sim.makeAction([0], [1.0], [0])

            state = sim.getState()[0]
            assert state.viewIndex == ix

            # get adjacent locations
            for j, loc in enumerate(state.navigableLocations[1:]):
                # if a loc is visible from multiple view, use the closest
                # view (in angular distance) as its representation
                distance = _loc_distance(loc)

                if (loc.viewpointId not in adj_dict or
                        distance < adj_dict[loc.viewpointId]['distance']):
                    adj_dict[loc.viewpointId] = {
                        'scanId': scanId,
                        'viewpointId': loc.viewpointId,  # Next viewpoint id
                        'pointId': ix,
                        'distance': distance,
                        'idx': j + 1,
                        'position': (loc.x, loc.y, loc.z),
                    }
        candidate = list(adj_dict.values())
        buffered_state_dict[long_id] = [
            {key: c[key]
             for key in
             ['scanId', 'viewpointId', 'pointId', 'distance', 'idx', 'position']}
            for c in candidate
        ]
        return candidate
    else:
        candidate = buffered_state_dict[long_id]
        candidate_new = []
        for c in candidate:
            c_new = c.copy()
            candidate_new.append(c_new)
        return candidate_new


def generate_candidate_labels():
    '''Generate a json file of all nodes adjacent to each node. '''
    with open('../datasets/labels/house_pano_info.json') as info_file:
        scans = json.load(info_file)
    keys_list = list(scans.keys())
    values_list = list(scans.values())
    for i, scan_id in enumerate(keys_list):
        tmp = {}
        for viewpoint in list(values_list[i].keys()):
            try:
                candidates = make_candidate(scan_id, viewpoint)
                next_viewpoint_list = []
                for cand in candidates:
                    next_viewpoint_list.append(cand['viewpointId'])
                tmp[viewpoint] = next_viewpoint_list
            except:
                continue
        all_candidate_label[scan_id] = tmp
    filename = '../datasets/labels/candidate_label.json'
    with open(filename, 'w') as f_obj:
        json.dump(all_candidate_label, f_obj)


