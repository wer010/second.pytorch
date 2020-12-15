import time
from collections import defaultdict
from collections import OrderedDict
import numpy as np
from second.builder import anchor_generator_builder
from second.core import box_np_ops
from second.core import preprocess as prep
from second.core.geometry import points_in_convex_polygon_3d_jit
from second.data import kitti_common as kitti


def merge_second_batch(batch_list):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points', 'num_voxels','num_gt', 'gt_names', 'gt_classes', 'gt_boxes'
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'metadata':
            ret[key] = elems
        elif key == 'anchors':
            ret[key] = elems[0]
        elif key =='targets':
            labels = []
            bbox_targets = []
            importance = []
            for elem in elems:
                labels.append(elem['labels'])
                bbox_targets.append(elem['bbox_targets'])
                importance.append(elem['importance'])
            if 'use_quadrant' in example_merged:
                ret['labels'] = np.concatenate(labels,axis=0)
                ret['bbox_targets'] = np.concatenate(bbox_targets, axis=0)
                ret['importance'] = np.concatenate(importance, axis=0)
            else:
                ret['labels'] = np.stack(labels, axis=0)
                ret['bbox_targets'] = np.stack(bbox_targets, axis=0)
                ret['importance'] = np.stack(importance, axis=0)

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

def split_gt_into_quadrants(gt_dict):

    gt_boxes, gt_names, gt_classes, gt_importance = gt_dict["gt_boxes"], \
                                                    gt_dict["gt_names"], \
                                                    gt_dict["gt_classes"], \
                                                    gt_dict["gt_importance"]
    quadrant_1st_mask = np.multiply(gt_boxes[:, 0] >= 0, gt_boxes[:, 1] >= 0)
    quadrant_2nd_mask = np.multiply(gt_boxes[:, 0] < 0, gt_boxes[:, 1] >= 0)
    quadrant_3rd_mask = np.multiply(gt_boxes[:, 0] < 0, gt_boxes[:, 1] < 0)
    quadrant_4th_mask = np.multiply(gt_boxes[:, 0] >= 0, gt_boxes[:, 1] < 0)

    n = np.sum(quadrant_1st_mask) + \
        np.sum(quadrant_2nd_mask) + \
        np.sum(quadrant_3rd_mask) + \
        np.sum(quadrant_4th_mask)
    assert gt_boxes.shape[0] == n, f'{gt_boxes.shape[0]} is not equal to {n}'
    gt_boxes_1st_quadrant = gt_boxes[quadrant_1st_mask, :]
    gt_names_1st_quadrant = gt_names[quadrant_1st_mask]
    gt_classes_1st_quadrant = gt_classes[quadrant_1st_mask]
    gt_importance_1st_quadrant = gt_importance[quadrant_1st_mask]

    gt_boxes_2nd_quadrant = gt_boxes[quadrant_2nd_mask, :]
    gt_boxes_2nd_quadrant[:, 0] = -gt_boxes_2nd_quadrant[:, 0]
    gt_boxes_2nd_quadrant[:, 6] = -gt_boxes_2nd_quadrant[:, 6]
    gt_names_2nd_quadrant = gt_names[quadrant_2nd_mask]
    gt_classes_2nd_quadrant = gt_classes[quadrant_2nd_mask]
    gt_importance_2nd_quadrant = gt_importance[quadrant_2nd_mask]


    gt_boxes_3rd_quadrant = gt_boxes[quadrant_3rd_mask, :]
    gt_boxes_3rd_quadrant[:, 0:2] = -gt_boxes_3rd_quadrant[:, 0:2]
    gt_boxes_3rd_quadrant[:, 6] = gt_boxes_3rd_quadrant[:, 6] + np.pi
    gt_names_3rd_quadrant = gt_names[quadrant_3rd_mask]
    gt_classes_3rd_quadrant = gt_classes[quadrant_3rd_mask]
    gt_importance_3rd_quadrant = gt_importance[quadrant_3rd_mask]


    gt_boxes_4th_quadrant = gt_boxes[quadrant_4th_mask, :]
    gt_boxes_4th_quadrant[:, 1] = -gt_boxes_4th_quadrant[:, 1]
    gt_boxes_4th_quadrant[:, 6] = -gt_boxes_4th_quadrant[:, 6] + np.pi
    gt_names_4th_quadrant = gt_names[quadrant_4th_mask]
    gt_classes_4th_quadrant = gt_classes[quadrant_4th_mask]
    gt_importance_4th_quadrant = gt_importance[quadrant_4th_mask]


    mapped_gt_boxes = [gt_boxes_1st_quadrant, gt_boxes_2nd_quadrant, gt_boxes_3rd_quadrant, gt_boxes_4th_quadrant]
    mapped_gt_names = [gt_names_1st_quadrant, gt_names_2nd_quadrant, gt_names_3rd_quadrant, gt_names_4th_quadrant]
    mapped_gt_classes = [gt_classes_1st_quadrant, gt_classes_2nd_quadrant, gt_classes_3rd_quadrant,
                         gt_classes_4th_quadrant]
    mapped_gt_importance = [gt_importance_1st_quadrant,gt_importance_2nd_quadrant,
                            gt_importance_3rd_quadrant,gt_importance_4th_quadrant]

    ret = {
        "gt_boxes":mapped_gt_boxes,
        "gt_names":mapped_gt_names,
        "gt_classes":mapped_gt_classes,
        "gt_importance":mapped_gt_importance
    }

    return ret

def split_voxel_into_quadrants(input, voxel_shape):
    # convert voxel data of the 2nd, 3rd, 4th quadrant into 1st quadrant
    # input: dict, contain voxels and coordinates, which represent the voxel data (n,60,4),at most 60 points per voxel and (x,y,z,I)
    # and the corresponding coordinates, shape(n,4),(batch, z, y, x)
    # voxel_shape: (h,w,d) denote the number of voxel grid in x,y,z direction
    # return: dict, contain voxels and coordinates, all points in the voxels map into 1st quadrant,
    # coordinates,shape(n,5), (batch , quadrant index, z, y, x), and the y, x are recalculate from center point
    voxels = input["voxels"]
    coors = input["coordinates"]

    y = coors[:, 1]
    x = coors[:, 2]
    quadrant_range = voxel_shape[0:2]//2
    quadrant_range = quadrant_range[::-1]
    # print(f'max is {np.max(coors[:,2:4])}, min is {np.min(coors[:,2:4])}')

    quadrant_1st_mask = np.multiply(x>=quadrant_range[0] , y>=quadrant_range[1])
    quadrant_2nd_mask = np.multiply(x < quadrant_range[0] , y >= quadrant_range[1])
    quadrant_3rd_mask = np.multiply(x<quadrant_range[0] , y<quadrant_range[1])
    quadrant_4th_mask = np.multiply(x >= quadrant_range[0] , y < quadrant_range[1])


    coors[quadrant_1st_mask,1:3]= coors[quadrant_1st_mask,1:3] - quadrant_range

    voxels[quadrant_2nd_mask, :, 0] *= -1
    coors[quadrant_2nd_mask,1]= coors[quadrant_2nd_mask,1] - quadrant_range[0]

    voxels[quadrant_3rd_mask, :, 0:2] *= -1

    voxels[quadrant_4th_mask,:,1] *=  -1
    coors[quadrant_4th_mask,2]= coors[quadrant_4th_mask,2] - quadrant_range[1]

    # test_v = input["voxels"]
    # test_v[:,:,0:2] = np.absolute(test_v[:,:,0:2])
    # assert np.allclose(voxels, test_v)
    # print(f'max is {np.max(voxels)}, min is {np.min(voxels)}')

    quadrant_index = quadrant_2nd_mask.astype(np.int) * 1 \
                     + quadrant_3rd_mask.astype(np.int) * 2 \
                     + quadrant_4th_mask.astype(np.int) * 3

    coors = np.insert(coors, 0, quadrant_index, 1)

    n = np.sum(quadrant_1st_mask)+\
           np.sum(quadrant_2nd_mask)+\
           np.sum(quadrant_3rd_mask)+\
           np.sum(quadrant_4th_mask)
    assert voxels.shape[0]== n, f'{voxels.shape[0]} is not equal to {n}'

    # print(f'max is {np.max(coors[:, 2:4])}, min is {np.min(coors[:, 2:4])}')
    input["voxels"] = voxels
    input["coordinates"] = coors
    return input



def generate_anchors(classes_cfg, feature_map_size, use_quadrant=False):
    classes = []
    anchor_generators = []
    feature_map_sizes = []
    anchors_dict = OrderedDict()

    for class_setting in classes_cfg:
        anchor_generator = anchor_generator_builder.build(class_setting)
        if anchor_generator is not None:
            anchor_generators.append(anchor_generator)
        classes.append(class_setting.class_name)
        feature_map_sizes.append(class_setting.feature_map_size)

    for a in anchor_generators:
        anchors_dict[a.class_name] = {}

    anchors_list = []
    ndim = len(feature_map_size)
    matched_thresholds = [
        a.match_threshold for a in anchor_generators
    ]
    unmatched_thresholds = [
        a.unmatch_threshold for a in anchor_generators
    ]
    match_list, unmatch_list = [], []

    idx = 0
    box_ndim = anchor_generators[0].ndim
    for anchor_generator, match_thresh, unmatch_thresh, fsize in zip(anchor_generators, matched_thresholds,
                                                                     unmatched_thresholds, feature_map_sizes):
        if len(fsize) == 0:
            fsize = feature_map_size
            feature_map_sizes[idx] = feature_map_size
        anchors = anchor_generator.generate(fsize)
        anchors = anchors.reshape([*fsize, -1, box_ndim])
        anchors = anchors.transpose(ndim, *range(0, ndim), ndim + 1)

        if use_quadrant:
            anchors = anchors[:,:,(fsize[1])//2:fsize[1],(fsize[2])//2:fsize[2],...]

        anchors_list.append(anchors.reshape(-1, box_ndim))
        num_anchors = np.prod(anchors.shape[:-1])
        match_list.append(np.full([num_anchors], match_thresh, anchors.dtype))
        unmatch_list.append(np.full([num_anchors], unmatch_thresh, anchors.dtype))

        class_name = anchor_generator.class_name
        anchors_dict[class_name]["anchors"] = anchors.reshape(-1, box_ndim)
        anchors_dict[class_name]["matched_thresholds"] = match_list[-1]
        anchors_dict[class_name]["unmatched_thresholds"] = unmatch_list[-1]
        idx += 1
    anchors = np.concatenate(anchors_list, axis=0)
    matched_thresholds = np.concatenate(match_list, axis=0)
    unmatched_thresholds = np.concatenate(unmatch_list, axis=0)

    return {"anchors": anchors,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
            "anchors_dict": anchors_dict}


def prep_pointcloud(input_dict,
                    root_path,
                    voxel_generator,
                    target_assigner,
                    use_quadrant=False,
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

    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // out_size_factor
    feature_map_size = [*feature_map_size, 1][::-1]
    # [352, 400]


    t1 = time.time()
    if not multi_gpu:
        res = voxel_generator.generate(points, max_voxels)
        voxels = res["voxels"]
        coordinates = res["coordinates"]
        num_points = res["num_points_per_voxel"]
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)
    else:
        res = voxel_generator.generate_multi_gpu(points, max_voxels)
        voxels = res["voxels"]
        coordinates = res["coordinates"]
        num_points = res["num_points_per_voxel"]
        num_voxels = np.array([res["voxel_num"]], dtype=np.int64)
    metrics["voxel_gene_time"] = time.time() - t1

    anchors_all = generate_anchors(target_assigner.classes_cfg, feature_map_size, use_quadrant)
    if use_quadrant:
        split_res = split_voxel_into_quadrants(res, grid_size)
        voxels = split_res["voxels"]
        coordinates = split_res["coordinates"]




    example = { 'voxels': voxels,
                'num_points': num_points,
                'coordinates': coordinates,
                "num_voxels": num_voxels,
                "metrics": metrics ,
                'anchors': anchors_all["anchors"]}
    if calib is not None:
        example["calib"] = calib


    metrics["prep_time"] = time.time() - t

    if not training:
        return example
    # voxel_labels = box_np_ops.assign_label_to_voxel(gt_boxes, coordinates,
    #                                                 voxel_size, coors_range)

    if create_targets:
        if target_assigner.name  == 'LabelAssigner':
            if use_quadrant:
                res = []
                gt_box = split_dict["gt_boxes"]
                gt_classes = split_dict["gt_classes"]
                gt_names = split_dict["gt_name"]
                gt_importance = split_dict["gt_importance"]
                for i in range(4):
                    d = target_assigner.assign(
                        gt_boxes=gt_box[i],
                        feature_map_size=feature_map_size,
                        gt_classes=gt_classes[i],
                        gt_names=gt_names[i],
                        importance=gt_importance[i],
                        training=training,
                        dataset_name=dataset_name
                    )
                    res.append(d)
            else:
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
            if use_quadrant:
                split_dict = split_gt_into_quadrants(gt_dict)
                example['use_quadrant']= use_quadrant
                targets_dict = {}
                gt_box = split_dict["gt_boxes"]
                gt_classes = split_dict["gt_classes"]
                gt_names = split_dict["gt_names"]
                gt_importance = split_dict["gt_importance"]
                labels=[]
                bbox_targets=[]
                importance = []
                for i in range(4):
                    d = target_assigner.assign(
                        anchors_all,
                        gt_boxes=gt_box[i],
                        gt_classes=gt_classes[i],
                        gt_names=gt_names[i],
                        importance=gt_importance[i]
                    )

                    labels.append(d['labels'])
                    bbox_targets.append(d['bbox_targets'])
                    importance.append(d['importance'])
                targets_dict['labels']=np.stack(labels)
                targets_dict['bbox_targets'] = np.stack(bbox_targets)
                targets_dict['importance'] = np.stack(importance)

            else:
                targets_dict = target_assigner.assign(
                    anchors_all,
                    gt_dict["gt_boxes"],
                    gt_classes=gt_dict["gt_classes"],
                    gt_names=gt_dict["gt_names"],
                    importance=gt_dict["gt_importance"])

            example.update({'targets': targets_dict})


    return example
