import pathlib
import pickle
import time
from collections import defaultdict
from functools import partial

import cv2
import numpy as np
from skimage import io as imgio

from second.core import box_np_ops
from second.core import preprocess as prep
from second.core.geometry import points_in_convex_polygon_3d_jit
from second.data import kitti_common as kitti
from second.utils import simplevis


def merge_second_batch(batch_list):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points', 'num_gt', 'voxel_labels', 'gt_names', 'gt_classes', 'gt_boxes'
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'metadata':
            ret[key] = elems
        elif key =='targets':
            cache = []
            for elem in elems:
                for e in elem:
                    cache.append(e)

            ret['labels'] = [elem['labels'] for elem in cache]
            ret['labels'] = np.stack(ret['labels'],axis=0)
            ret['bbox_targets'] = [elem['bbox_targets'] for elem in cache]
            ret['bbox_targets'] = np.stack(ret['bbox_targets'], axis=0)
            ret['importance'] = [elem['importance'] for elem in cache]
            ret['importance'] = np.stack(ret['importance'], axis=0)
            ret['gt_names'] = [elem['gt_names'] for elem in cache]
            ret['gt_classes'] = [elem['gt_classes'] for elem in cache]

        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = np.stack(v1, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        elif key == 'metrics':
            ret[key] = elems
        elif key in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "hm", "anno_box",
                    "ind", "mask", "cat"]:
            ret[key] = defaultdict(list)
            res = []
            for elem in elems:
                for idx, ele in enumerate(elem):
                    ret[key][str(idx)].append(ele)
            for kk, vv in ret[key].items():
                res.append(np.stack(vv))
            ret[key] = res
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret

def merge_second_batch_multigpu(batch_list):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if key == 'metadata':
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
                ret[key][k1] = np.stack(v1, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.stack(coors, axis=0)
        elif key in ['gt_names', 'gt_classes', 'gt_boxes']:
            continue
        else:
            ret[key] = np.stack(elems, axis=0)
        
    return ret


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def prep_pointcloud(input_dict,
                    root_path,
                    voxel_generator,
                    target_assigner,
                    db_sampler=None,
                    max_voxels=20000,
                    remove_outside_points=False,
                    training=True,
                    create_targets=True,
                    shuffle_points=False,
                    remove_unknown=False,
                    gt_rotation_noise=(-np.pi / 3, np.pi / 3),
                    gt_loc_noise_std=(1.0, 1.0, 1.0),
                    global_rotation_noise=(-np.pi / 4, np.pi / 4),
                    global_scaling_noise=(0.95, 1.05),
                    global_random_rot_range=(0.78, 2.35),
                    global_translate_noise_std=(0, 0, 0),
                    num_point_features=4,
                    anchor_area_threshold=1,
                    gt_points_drop=0.0,
                    gt_drop_max_keep=10,
                    remove_points_after_sample=True,
                    remove_environment=False,
                    random_crop=False,
                    reference_detections=None,
                    out_size_factor=2,
                    use_group_id=False,
                    multi_gpu=False,
                    min_points_in_gt=-1,
                    random_flip_x=True,
                    random_flip_y=True,
                    sample_importance=1.0,
                    out_dtype=np.float32,
                    dataset_name = 'KITTI'):
    """convert point cloud to voxels, create targets if ground truths 
    exists.

    input_dict format: dataset.get_sensor_data format

    """
    t = time.time()
    class_names = target_assigner.classes
    points = input_dict["lidar"]["points"]

    if training:
        anno_dict = input_dict["lidar"]["annotations"]
        gt_dict = {
            "gt_boxes": anno_dict["boxes"],
            "gt_names": anno_dict["names"],
            "gt_importance": np.ones([anno_dict["boxes"].shape[0]], dtype=anno_dict["boxes"].dtype),
        }
        if "difficulty" not in anno_dict:
            difficulty = np.zeros([anno_dict["boxes"].shape[0]],
                                  dtype=np.int32)
            gt_dict["difficulty"] = difficulty
        else:
            gt_dict["difficulty"] = anno_dict["difficulty"]
        if use_group_id and "group_ids" in anno_dict:
            group_ids = anno_dict["group_ids"]
            gt_dict["group_ids"] = group_ids
    calib = None
    if "calib" in input_dict:
        calib = input_dict["calib"]

    if reference_detections is not None:
        assert calib is not None and "image" in input_dict
        C, R, T = box_np_ops.projection_matrix_to_CRT_kitti(P2)
        frustums = box_np_ops.get_frustum_v2(reference_detections, C)
        frustums -= T
        frustums = np.einsum('ij, akj->aki', np.linalg.inv(R), frustums)
        frustums = box_np_ops.camera_to_lidar(frustums, rect, Trv2c)
        surfaces = box_np_ops.corner_to_surfaces_3d_jit(frustums)
        masks = points_in_convex_polygon_3d_jit(points, surfaces)
        points = points[masks.any(-1)]

    if remove_outside_points:
        assert calib is not None
        image_shape = input_dict["image"]["image_shape"]
        points = box_np_ops.remove_outside_points(
            points, calib["rect"], calib["Trv2c"], calib["P2"], image_shape)
    if remove_environment and training:
        selected = kitti.keep_arrays_by_name(gt_names, target_assigner.classes)
        _dict_select(gt_dict, selected)
        masks = box_np_ops.points_in_rbbox(points, gt_dict["gt_boxes"])
        points = points[masks.any(-1)]
    metrics = {}

    if training:
        # boxes_lidar = gt_dict["gt_boxes"]
        # bev_map = simplevis.nuscene_vis(points, boxes_lidar)
        # cv2.imshow('pre-noise', bev_map)
        selected = kitti.drop_arrays_by_name(gt_dict["gt_names"], ["DontCare"])
        _dict_select(gt_dict, selected)
        if remove_unknown:
            remove_mask = gt_dict["difficulty"] == -1
            """
            gt_boxes_remove = gt_boxes[remove_mask]
            gt_boxes_remove[:, 3:6] += 0.25
            points = prep.remove_points_in_boxes(points, gt_boxes_remove)
            """
            keep_mask = np.logical_not(remove_mask)
            _dict_select(gt_dict, keep_mask)
        gt_dict.pop("difficulty")
        if min_points_in_gt > 0:
            # points_count_rbbox takes 10ms with 10 sweeps nuscenes data
            point_counts = box_np_ops.points_count_rbbox(points, gt_dict["gt_boxes"])
            mask = point_counts >= min_points_in_gt
            _dict_select(gt_dict, mask)
        # select the gt_box in the specified classes
        gt_boxes_mask = np.array(
            [n in class_names for n in gt_dict["gt_names"]], dtype=np.bool_)
        # data augmentation
        if db_sampler is not None:
            group_ids = None
            if "group_ids" in gt_dict:
                group_ids = gt_dict["group_ids"]

            # data augmentation, using sample to add target
            sampled_dict = db_sampler.sample_all(
                root_path,
                gt_dict["gt_boxes"],
                gt_dict["gt_names"],
                num_point_features,
                random_crop,
                gt_group_ids=group_ids,
                calib=calib)

            if sampled_dict is not None:
                sampled_gt_names = sampled_dict["gt_names"]
                sampled_gt_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                sampled_gt_masks = sampled_dict["gt_masks"]
                gt_dict["gt_names"] = np.concatenate(
                    [gt_dict["gt_names"], sampled_gt_names], axis=0)
                gt_dict["gt_boxes"] = np.concatenate(
                    [gt_dict["gt_boxes"], sampled_gt_boxes])
                gt_boxes_mask = np.concatenate(
                    [gt_boxes_mask, sampled_gt_masks], axis=0)
                sampled_gt_importance = np.full([sampled_gt_boxes.shape[0]], sample_importance, dtype=sampled_gt_boxes.dtype)
                gt_dict["gt_importance"] = np.concatenate(
                    [gt_dict["gt_importance"], sampled_gt_importance])

                if group_ids is not None:
                    sampled_group_ids = sampled_dict["group_ids"]
                    gt_dict["group_ids"] = np.concatenate(
                        [gt_dict["group_ids"], sampled_group_ids])
                # remove the raw points in the added box, to avoid overlap
                if remove_points_after_sample:
                    masks = box_np_ops.points_in_rbbox(points,
                                                       sampled_gt_boxes)
                    points = points[np.logical_not(masks.any(-1))]

                points = np.concatenate([sampled_points, points], axis=0)
        group_ids = None
        if "group_ids" in gt_dict:
            group_ids = gt_dict["group_ids"]

        prep.noise_per_object_v3_(gt_dict["gt_boxes"],
                                    points,
                                    gt_boxes_mask,
                                    rotation_perturb=gt_rotation_noise,
                                    center_noise_std=gt_loc_noise_std,
                                    global_random_rot_range=global_random_rot_range,
                                    group_ids=group_ids,
                                    num_try=100)

        # should remove unrelated objects after noise per object
        # for k, v in gt_dict.items():
        #     print(k, v.shape)
        _dict_select(gt_dict, gt_boxes_mask)
        gt_classes = np.array(
            [class_names.index(n) + 1 for n in gt_dict["gt_names"]],
            dtype=np.int32)
        gt_dict["gt_classes"] = gt_classes
        gt_dict["gt_boxes"], points = prep.random_flip(gt_dict["gt_boxes"],
                                                       points, 0.5, random_flip_x, random_flip_y)
        gt_dict["gt_boxes"], points = prep.global_rotation_v2(
            gt_dict["gt_boxes"], points, *global_rotation_noise)
        gt_dict["gt_boxes"], points = prep.global_scaling_v2(
            gt_dict["gt_boxes"], points, *global_scaling_noise)
        prep.global_translate_(gt_dict["gt_boxes"], points, global_translate_noise_std)
        bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
        mask = prep.filter_gt_box_outside_range_by_center(gt_dict["gt_boxes"], bv_range)
        _dict_select(gt_dict, mask)

        # limit rad to [-pi, pi]
        gt_dict["gt_boxes"][:, 6] = box_np_ops.limit_period(
            gt_dict["gt_boxes"][:, 6], offset=0.5, period=2 * np.pi)

        # boxes_lidar = gt_dict["gt_boxes"]
        # bev_map = simplevis.nuscene_vis(points, boxes_lidar)
        # cv2.imshow('post-noise', bev_map)
        # cv2.waitKey(0)
    if shuffle_points:
        # shuffle is a little slow.
        np.random.shuffle(points)

    # [0, -40, -3, 70.4, 40, 1]
    voxel_size = voxel_generator.voxel_size
    pc_range = voxel_generator.point_cloud_range
    grid_size = voxel_generator.grid_size
    # [352, 400]
    t1 = time.time()
    if not multi_gpu:
        res = voxel_generator.generate(
            points, max_voxels)
        voxels = res["voxels"]
        coordinates = res["coordinates"]
        num_points = res["num_points_per_voxel"]
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
    else:
        res = voxel_generator.generate_multi_gpu(
            points, max_voxels)
        voxels = res["voxels"]
        coordinates = res["coordinates"]
        num_points = res["num_points_per_voxel"]
        num_voxels = np.array([res["voxel_num"]], dtype=np.int64)
    metrics["voxel_gene_time"] = time.time() - t1
    example = {
        'voxels': voxels,
        'num_points': num_points,
        'coordinates': coordinates,
        "num_voxels": num_voxels,
        "metrics": metrics,
    }
    if calib is not None:
        example["calib"] = calib
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]

    metrics["prep_time"] = time.time() - t
    if not training:
        return example
    # voxel_labels = box_np_ops.assign_label_to_voxel(gt_boxes, coordinates,
    #                                                 voxel_size, coors_range)
    if create_targets:
        t1 = time.time()
        if target_assigner.name  == 'LabelAssigner':
            targets_dict = target_assigner.assign(
                gt_dict["gt_boxes"],
                feature_map_size,
                gt_classes=gt_dict["gt_classes"],
                gt_names=gt_dict["gt_names"],
                importance=gt_dict["gt_importance"],
                training=training,
                dataset_name= dataset_name
            )
            example.update({
                'gt_dict': targets_dict['gt_dict'],
                'hm': targets_dict['targets']['hm'],
                'anno_box': targets_dict['targets']['anno_box'],
                'ind': targets_dict['targets']['ind'],
                'mask': targets_dict['targets']['mask'],
                'cat': targets_dict['targets']['cat']
            })
        else:
            targets_dict = target_assigner.assign(
                gt_dict["gt_boxes"],
                feature_map_size,
                gt_classes=gt_dict["gt_classes"],
                gt_names=gt_dict["gt_names"],
                importance=gt_dict["gt_importance"])
            example.update({
                'targets': targets_dict
            })


    return example
