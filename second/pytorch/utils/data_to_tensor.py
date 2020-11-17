import collections
from collections import defaultdict
import torch
import numpy as np


def collate_kitti(batch_list, device=None):
    device = device or torch.device("cuda:0")
    if 'cuda' in device.type:
        from multiprocessing import set_start_method
        try:
            set_start_method('spawn')
        except RuntimeError:
            pass
    example_merged = collections.defaultdict(list)
    for example in batch_list:
        if type(example) is list:
            for subexample in example:
                for k, v in subexample.items():
                    example_merged[k].append(v)
        else:
            for k, v in example.items():
                example_merged[k].append(v)
    batch_size = len(example_merged['metadata'])
    ret = {}

    for key, elems in example_merged.items():
        if key in ["voxels", "num_points", "num_gt", "voxel_labels", "num_voxels"]:
            ret[key] = torch.tensor(np.concatenate(elems, axis=0), device=device)
        elif key in ["gt_boxes"]:
            task_max_gts = []
            for task_id in range(len(elems[0])):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, len(elems[k][task_id]))
                task_max_gts.append(max_gt)
            res = []
            for idx, max_gt in enumerate(task_max_gts):
                batch_task_gt_boxes3d = np.zeros((batch_size, max_gt, 7))
                for i in range(batch_size):
                    batch_task_gt_boxes3d[i, : len(elems[i][idx]), :] = elems[i][idx]
                res.append(batch_task_gt_boxes3d)
            ret[key] = res
        elif key in ['metrics','gt_dict','metadata','gt_names']:
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = torch.tensor(np.stack(v1, axis=0), device=device)
        elif key in ["coordinates", "points"]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret[key] = torch.tensor(np.concatenate(coors, axis=0), device=device)
        elif key in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "hm", "anno_box",
                    "ind", "mask", "cat"]:
            ret[key] = defaultdict(list)
            res = []
            for elem in elems:
                for idx, ele in enumerate(elem):
                    ret[key][str(idx)].append(torch.tensor(ele), device=device)
            for kk, vv in ret[key].items():
                res.append(torch.stack(vv))
            ret[key] = res
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret


def cut_quadrant(input, voxel_shape):
    # convert voxel data of the 2nd, 3rd, 4th quadrant into 1st quadrant
    # input: dict, contain voxels and coordinates, which represent the voxel data (n,60,4),at most 60 points per voxel and (x,y,z,I)
    # and the corresponding coordinates, shape(n,4),(batch, z, y, x)
    # voxel_shape: (h,w,d) denote the number of voxel grid in x,y,z direction
    # return: dict, contain voxels and coordinates, all points in the voxels map into 1st quadrant,
    # coordinates,shape(n,5), (batch , quadrant index, z, y, x), and the y, x are recalculate from center point
    voxels = input["voxels"]
    coors = input["coordinates"]

    y = coors[:, 2]
    x = coors[:, 3]
    quadrant_range = voxel_shape//2
    # print(f'max is {np.max(coors[:,2:4])}, min is {np.min(coors[:,2:4])}')

    quadrant_1st_mask = np.multiply(x>=quadrant_range[0] , y>=quadrant_range[1])
    quadrant_2nd_mask = np.multiply(x < quadrant_range[0] , y >= quadrant_range[1])
    quadrant_3rd_mask = np.multiply(x<quadrant_range[0] , y<quadrant_range[1])
    quadrant_4th_mask = np.multiply(x >= quadrant_range[0] , y < quadrant_range[1])


    coors[quadrant_1st_mask,2:4]= coors[quadrant_1st_mask,2:4] - quadrant_range[0:2]

    voxels[quadrant_2nd_mask, :, 0] *= -1
    coors[quadrant_2nd_mask,2]= coors[quadrant_2nd_mask,2] - quadrant_range[1]

    voxels[quadrant_3rd_mask, :, 0:2] *= -1

    voxels[quadrant_4th_mask,:,1] *=  -1
    coors[quadrant_4th_mask,3]= coors[quadrant_4th_mask,3] - quadrant_range[0]

    # test_v = input["voxels"]
    # test_v[:,:,0:2] = np.absolute(test_v[:,:,0:2])
    # assert np.allclose(voxels, test_v)
    # print(f'max is {np.max(voxels)}, min is {np.min(voxels)}')

    quadrant_index = quadrant_2nd_mask.astype(np.int) * 1 \
                     + quadrant_3rd_mask.astype(np.int) * 2 \
                     + quadrant_4th_mask.astype(np.int) * 3

    coors = np.insert(coors, 1, quadrant_index, 1)

    n = np.sum(quadrant_1st_mask)+\
           np.sum(quadrant_2nd_mask)+\
           np.sum(quadrant_3rd_mask)+\
           np.sum(quadrant_4th_mask)
    assert voxels.shape[0]== n, f'{voxels.shape[0]} is not equal to {n}'

    # print(f'max is {np.max(coors[:, 2:4])}, min is {np.min(coors[:, 2:4])}')
    input["voxels"] = voxels
    input["coordinates"] = coors
    return input

def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "importance",'bbox_targets'
    ]
    center_point_target_name = ['hm', 'anno_box', 'ind', 'mask', 'cat']
    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib
        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v)
        elif k in center_point_target_name:
            res = []
            for item in v:
                res.append(torch.tensor(item, device=device))
            example_torch[k] = res
        else:
            example_torch[k] = v
    return example_torch

