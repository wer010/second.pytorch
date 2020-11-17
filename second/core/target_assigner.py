import numpy as np
from collections import OrderedDict

from second.core import box_np_ops, region_similarity
from second.core.target_ops import create_target_np
from second.utils.config_tool import get_downsample_factor
from second.protos import target_pb2, anchors_pb2
from second.builder import similarity_calculator_builder
from second.builder import anchor_generator_builder
from second.utils.center_utils import draw_umich_gaussian, gaussian_radius

class LabelAssigner():
    def __init__(self, model_cfg, box_coder):
        self.target_assigner_config = model_cfg.target_assigner
        if not isinstance(self.target_assigner_config, (target_pb2.TargetAssigner)):
            raise ValueError('input_reader_config not of type '
                             'input_reader_pb2.InputReader.')
        self.classes_cfg = self.target_assigner_config.class_settings
        self.tasks = [{'num_class': 1, 'class_names': ['car']},
                      {'num_class': 2, 'class_names': ['truck', 'construction_vehicle']},
                      {'num_class': 2, 'class_names': ['bus', 'trailer']}, {'num_class': 1, 'class_names': ['barrier']},
                      {'num_class': 2, 'class_names': ['motorcycle', 'bicycle']},
                      {'num_class': 2, 'class_names': ['pedestrian', 'traffic_cone']}]
        classes = [class_setting.class_name for class_setting in self.classes_cfg]


        self.similarity_calcs = []
        for class_setting in self.classes_cfg:
            self.similarity_calcs.append(similarity_calculator_builder.build(
                class_setting.region_similarity_calculator))

        self.positive_fraction = self.target_assigner_config.sample_positive_fraction
        if self.positive_fraction < 0:
            positive_fraction = None


        self.out_size_factor = get_downsample_factor(model_cfg)
        self.pc_range_start_end = np.asarray(model_cfg.voxel_generator.point_cloud_range)
        pc_range = self.pc_range_start_end[3:6] - self.pc_range_start_end[0:3]

        grid_size = model_cfg.voxel_generator.voxel_size
        self.voxel_shape = pc_range // grid_size

        feature_map_size = self.voxel_shape[:2] // self.out_size_factor
        self._feature_map_size = [*feature_map_size, 1][::-1]
        self._box_coder = box_coder
        self._grid_size = grid_size
        self._pc_range = pc_range
        self.name = 'LabelAssigner'

        self._sim_calcs = self.similarity_calcs
        self._positive_fraction = positive_fraction
        self._sample_size = self.target_assigner_config.sample_size
        self._classes = classes
        self._assign_per_class = self.target_assigner_config.assign_per_class

    def assign(self,
               gt_boxes,
               gt_classes=None,
               gt_names=None,
               importance=None,
               training=True,
               dataset_name = None):
        max_objs = 500
        # can't not add the tasks into config file, need some protobuf modify.

        class_names_by_task = [t['class_names'] for t in self.tasks]


        # Calculate output featuremap size
        grid_size = self.voxel_shape  # 448 x 512
        pc_range =  self.pc_range_start_end
        voxel_size = self._grid_size

        feature_map_size = self._feature_map_size[1:3]
        example = {}

        if training:
            gt_dict = {'gt_boxes':gt_boxes,
                       'gt_names':gt_names,
                       'gt_classes':gt_classes}

            # reorganize the gt_dict by tasks
            task_masks = []
            flag = 0
            for class_name in class_names_by_task:
                # print("classes: ", gt_dict["gt_classes"], "name", class_name)
                task_masks.append(
                    [
                        np.where(
                            gt_dict["gt_classes"] == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            task_names = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                for m in mask:
                    task_box.append(gt_dict["gt_boxes"][m])
                    task_class.append(gt_dict["gt_classes"][m] - flag2)
                    task_name.append(gt_dict["gt_names"][m])
                task_boxes.append(np.concatenate(task_box, axis=0))
                task_classes.append(np.concatenate(task_class))
                task_names.append(np.concatenate(task_name))
                flag2 += len(mask)

            for task_box in task_boxes:
                # limit rad to [-pi, pi]
                task_box[:, -1] = box_np_ops.limit_period(
                    task_box[:, -1], offset=0.5, period=np.pi * 2
                )

            # print(gt_dict.keys())
            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            draw_gaussian = draw_umich_gaussian
            hms, anno_boxs, inds, masks, cats = [], [], [], [], []
            for idx, task in enumerate(self.tasks):
                hm = np.zeros((len(class_names_by_task[idx]), int(feature_map_size[1]), int(feature_map_size[0])),
                              dtype=np.float32)

                if dataset_name == 'NuScenesDataset':
                    # [reg, hei, dim, vx, vy, rots, rotc]
                    anno_box = np.zeros((max_objs, 8), dtype=np.float32)
                elif dataset_name == 'WaymoDataset':
                    anno_box = np.zeros((max_objs, 8), dtype=np.float32)
                else:
                    raise NotImplementedError("Only Support nuScene for Now!")

                ind = np.zeros((max_objs), dtype=np.int64)
                mask = np.zeros((max_objs), dtype=np.uint8)
                cat = np.zeros((max_objs), dtype=np.int64)
                direction = np.zeros((max_objs), dtype=np.int64)

                num_objs = min(gt_dict['gt_boxes'][idx].shape[0], max_objs)
                for k in range(num_objs):
                    cls_id = gt_dict['gt_classes'][idx][k] - 1

                    w, l, h = gt_dict['gt_boxes'][idx][k][3], gt_dict['gt_boxes'][idx][k][4], \
                              gt_dict['gt_boxes'][idx][k][5]
                    w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                    if w > 0 and l > 0:
                        radius = gaussian_radius((l, w), min_overlap=0.1)
                        radius = max(2, int(radius))

                        # be really careful for the coordinate system of your box annotation.
                        x, y, z = gt_dict['gt_boxes'][idx][k][0], gt_dict['gt_boxes'][idx][k][1], \
                                  gt_dict['gt_boxes'][idx][k][2]

                        coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                         (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                        ct = np.array(
                            [coor_x, coor_y], dtype=np.float32)
                        ct_int = ct.astype(np.int32)

                        # throw out not in range objects to avoid out of array area when creating the heatmap
                        if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                            continue

                        draw_gaussian(hm[cls_id], ct, radius)

                        new_idx = k
                        x, y = ct_int[0], ct_int[1]

                        if not (y * feature_map_size[0] + x < feature_map_size[0] * feature_map_size[1]):
                            # a double check, should never happen
                            print(x, y, y * feature_map_size[0] + x)
                            assert False

                        cat[new_idx] = cls_id
                        ind[new_idx] = y * feature_map_size[0] + x
                        mask[new_idx] = 1

                        if dataset_name == 'NuScenesDataset':
                            # vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][6]
                            no_log = False
                            if not no_log:
                                anno_box[new_idx] = np.concatenate(
                                    (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                     np.sin(rot), np.cos(rot)), axis=None)
                            else:
                                anno_box[new_idx] = np.concatenate(
                                    (ct - (x, y), z, gt_dict['gt_boxes'][idx][k][3:6],
                                     np.sin(rot), np.cos(rot)), axis=None)
                        elif dataset_name == 'WaymoDataset':
                            rot = gt_dict['gt_boxes'][idx][k][-1]
                            anno_box[new_idx] = np.concatenate(
                                (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                np.sin(rot), np.cos(rot)), axis=None)

                        else:
                            raise NotImplementedError("Only Support KITTI and nuScene for Now!")

                hms.append(hm)
                anno_boxs.append(anno_box)
                masks.append(mask)
                inds.append(ind)
                cats.append(cat)

            example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})
        else:
            pass

        targets_dict = {'gt_dict':gt_dict, "targets": example}

        return targets_dict

    @property
    def box_coder(self):
        return self._box_coder
    @property
    def classes(self):
        return self._classes

# TODO: something goes wrong, but not big deal
class TargetAssigner(LabelAssigner):
    def __init__(self, model_cfg, box_coder):
        super().__init__(model_cfg, box_coder)
        anchor_generators = []
        classes = []
        feature_map_sizes = []
        for class_setting in self.classes_cfg:
            anchor_generator = anchor_generator_builder.build(class_setting)
            if anchor_generator is not None:
                anchor_generators.append(anchor_generator)
            else:
                assert self.target_assigner_config.assign_per_class is False
            classes.append(class_setting.class_name)
            feature_map_sizes.append(class_setting.feature_map_size)
        self._box_coder = box_coder
        self._anchor_generators = anchor_generators
        box_ndims = [a.ndim for a in anchor_generators]
        assert all([e == box_ndims[0] for e in box_ndims])
        self._feature_map_sizes = feature_map_sizes
        self.name = 'TargetAssigner'
        self.use_quadrant = model_cfg.use_quadrant

    def generate_anchor_mask(self, anchors, anchor_area_threshold=-1):
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
        anchors_mask = None
        if anchor_area_threshold >= 0:
            # slow with high resolution. recommend disable this forever.
            coors = coordinates
            dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
                coors, tuple(grid_size[::-1][1:]))
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            anchors_area = box_np_ops.fused_get_anchors_area(
                dense_voxel_map, anchors_bv, voxel_size, pc_range, grid_size)
            anchors_mask = anchors_area > anchor_area_threshold
            # example['anchors_mask'] = anchors_mask.astype(np.uint8)
        return anchors_mask



    def assign(self,
               gt_boxes,
               feature_map_size,
               gt_classes=None,
               gt_names=None,
               importance=None):
        if self.use_quadrant:
            feature_map_size[1] = feature_map_size[1]//2
            feature_map_size[2] = feature_map_size[2] // 2


            quadrant_1st_mask = np.multiply(gt_boxes[:,0] >= 0, gt_boxes[:,1] >= 0)
            quadrant_2nd_mask = np.multiply(gt_boxes[:,0] < 0, gt_boxes[:,1] >= 0)
            quadrant_3rd_mask = np.multiply(gt_boxes[:,0] < 0, gt_boxes[:,1] < 0)
            quadrant_4th_mask = np.multiply(gt_boxes[:,0] >= 0, gt_boxes[:,1] < 0)

            n = np.sum(quadrant_1st_mask) + \
                np.sum(quadrant_2nd_mask) + \
                np.sum(quadrant_3rd_mask) + \
                np.sum(quadrant_4th_mask)
            assert gt_boxes.shape[0] == n, f'{gt_boxes.shape[0]} is not equal to {n}'
            gt_boxes_1st_quadrant = gt_boxes[quadrant_1st_mask, :]
            gt_names_1st_quadrant = gt_names[quadrant_1st_mask]
            gt_classes_1st_quadrant = gt_classes[quadrant_1st_mask]

            gt_boxes_2nd_quadrant = gt_boxes[quadrant_2nd_mask, :]
            gt_boxes_2nd_quadrant[:,0] = -gt_boxes_2nd_quadrant[:,0]
            gt_boxes_2nd_quadrant[:, 6] = -gt_boxes_2nd_quadrant[:, 6]
            gt_names_2nd_quadrant = gt_names[quadrant_2nd_mask]
            gt_classes_2nd_quadrant = gt_classes[quadrant_2nd_mask]

            gt_boxes_3rd_quadrant = gt_boxes[quadrant_3rd_mask, :]
            gt_boxes_3rd_quadrant[:,0:2] = -gt_boxes_3rd_quadrant[:,0:2]
            gt_boxes_3rd_quadrant[:, 6] = gt_boxes_3rd_quadrant[:, 6] + np.pi
            gt_names_3rd_quadrant = gt_names[quadrant_3rd_mask]
            gt_classes_3rd_quadrant = gt_classes[quadrant_3rd_mask]


            gt_boxes_4th_quadrant = gt_boxes[quadrant_4th_mask, :]
            gt_boxes_4th_quadrant[:,1] = -gt_boxes_4th_quadrant[:,1]
            gt_boxes_4th_quadrant[:, 6] = -gt_boxes_4th_quadrant[:, 6] + np.pi
            gt_names_4th_quadrant = gt_names[quadrant_4th_mask]
            gt_classes_4th_quadrant = gt_classes[quadrant_4th_mask]

            mapped_gt_boxes = [gt_boxes_1st_quadrant,gt_boxes_2nd_quadrant,gt_boxes_3rd_quadrant,gt_boxes_4th_quadrant]
            mapped_gt_names = [gt_names_1st_quadrant,gt_names_2nd_quadrant,gt_names_3rd_quadrant,gt_names_4th_quadrant]
            mapped_gt_classes = [gt_classes_1st_quadrant,gt_classes_2nd_quadrant,gt_classes_3rd_quadrant,gt_classes_4th_quadrant]
        if self._assign_per_class:
            anchors_dict = self.generate_anchors_dict(feature_map_size)
            if self.use_quadrant:
                res = []
                for gt_box, gt_class, gt_name in zip(mapped_gt_boxes,mapped_gt_names,mapped_gt_classes):
                    d = self.assign_per_class(anchors_dict, gt_box, gt_class, gt_name, importance=importance,anchors_mask=None)
                    d['gt_names'] = gt_name
                    d['gt_classes'] = gt_class
                    res.append(d)
                return res
            else:
                return self.assign_per_class(anchors_dict, gt_boxes, gt_classes, gt_names, importance=importance,anchors_mask=None)
        else:
            ret = self.generate_anchors(feature_map_size)
            anchors = ret['anchors'].reshape(-1, self.box_ndim)
            matched_thresholds = ret["matched_thresholds"]
            unmatched_thresholds = ret["unmatched_thresholds"]
            return self.assign_all(anchors, gt_boxes, gt_classes, matched_thresholds, unmatched_thresholds, importance=importance,anchors_mask=None)

    def assign_all(self,
                   anchors,
                   gt_boxes,
                   gt_classes=None,
                   matched_thresholds=None,
                   unmatched_thresholds=None,
                   importance=None,
                   anchors_mask=None):
        if anchors_mask is not None:
            prune_anchor_fn = lambda _: np.where(anchors_mask)[0]
        else:
            prune_anchor_fn = None

        def similarity_fn(anchors, gt_boxes):
            anchors_rbv = anchors[:, [0, 1, 3, 4, 6]]
            gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
            return self._sim_calcs[0].compare(anchors_rbv, gt_boxes_rbv)

        def box_encoding_fn(boxes, anchors):
            return self._box_coder.encode(boxes, anchors)
        return create_target_np(
            anchors,
            gt_boxes,
            similarity_fn,
            box_encoding_fn,
            prune_anchor_fn=prune_anchor_fn,
            gt_classes=gt_classes,
            matched_threshold=matched_thresholds,
            unmatched_threshold=unmatched_thresholds,
            positive_fraction=self._positive_fraction,
            rpn_batch_size=self._sample_size,
            norm_by_num_examples=False,
            box_code_size=self.box_coder.code_size,
            gt_importance=importance)

    def assign_per_class(self,
                         anchors_dict,
                         gt_boxes,
                         gt_classes=None,
                         gt_names=None,
                         importance=None,
                         anchors_mask=None):
        """this function assign target individally for each class.
        recommend for multi-class network.
        """

        def box_encoding_fn(boxes, anchors):
            return self._box_coder.encode(boxes, anchors)

        def similarity_fn(anchors, gt_boxes):
            anchors_rbv = anchors[:, [0, 1, 3, 4, 6]]
            gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
            return self._sim_calcs[anchor_gene_idx].compare(
                anchors_rbv, gt_boxes_rbv)

        targets_list = []
        anchor_loc_idx = 0
        anchor_gene_idx = 0
        for class_name, anchor_dict in anchors_dict.items():

            mask = np.array([c == class_name for c in gt_names], dtype=np.bool_)
            num_loc = anchor_dict["anchors"].shape[-2]
            if anchors_mask is not None:
                anchors_mask = anchors_mask.reshape(-1)
                a_range = self.anchors_range(class_name)
                anchors_mask_class = anchors_mask[a_range[0]:a_range[1]].reshape(-1)
                prune_anchor_fn = lambda _: np.where(anchors_mask_class)[0]
            else:
                prune_anchor_fn = None
            # print(f"num of {class_name}:", np.sum(mask))
            targets = create_target_np(
                anchor_dict["anchors"].reshape(-1, self.box_ndim),
                gt_boxes[mask],
                similarity_fn,
                box_encoding_fn,
                prune_anchor_fn=prune_anchor_fn,
                gt_classes=gt_classes[mask],
                matched_threshold=anchor_dict["matched_thresholds"],
                unmatched_threshold=anchor_dict["unmatched_thresholds"],
                positive_fraction=self._positive_fraction,
                rpn_batch_size=self._sample_size,
                norm_by_num_examples=False,
                box_code_size=self.box_coder.code_size,
                gt_importance=importance)
            # print(f"num of positive:", np.sum(targets["labels"] == self.classes.index(class_name) + 1))
            anchor_loc_idx += num_loc
            targets_list.append(targets)
            anchor_gene_idx += 1

        targets_dict = {"labels": [t["labels"] for t in targets_list],
            "bbox_targets": [t["bbox_targets"] for t in targets_list],
            "importance": [t["importance"] for t in targets_list]}
        targets_dict["bbox_targets"] = np.concatenate(
            [v.reshape(-1, self.box_coder.code_size)for v in targets_dict["bbox_targets"]], axis=0)
        targets_dict["bbox_targets"] = targets_dict["bbox_targets"].reshape(-1, self.box_coder.code_size)
        targets_dict["labels"] = np.concatenate([v.reshape(-1) for v in targets_dict["labels"]], axis=0)
        targets_dict["importance"] = np.concatenate([v.reshape(-1) for v in targets_dict["importance"]], axis=0)
        targets_dict["labels"] = targets_dict["labels"].reshape(-1)
        targets_dict["importance"] = targets_dict["importance"].reshape(-1)

        return targets_dict

    def generate_anchors(self, feature_map_size):
        anchors_list = []
        ndim = len(feature_map_size)
        matched_thresholds = [
            a.match_threshold for a in self._anchor_generators
        ]
        unmatched_thresholds = [
            a.unmatch_threshold for a in self._anchor_generators
        ]
        match_list, unmatch_list = [], []
        if self._feature_map_sizes is not None:
            feature_map_sizes = self._feature_map_sizes
        else:
            feature_map_sizes = [feature_map_size] * len(self._anchor_generators)
        idx = 0
        for anchor_generator, match_thresh, unmatch_thresh, fsize in zip(
                self._anchor_generators, matched_thresholds,
                unmatched_thresholds, feature_map_sizes):
            if len(fsize) == 0:
                fsize = feature_map_size
                self._feature_map_sizes[idx] = feature_map_size
            anchors = anchor_generator.generate(fsize)
            anchors = anchors.reshape([*fsize, -1, self.box_ndim])
            anchors = anchors.transpose(ndim, *range(0, ndim), ndim + 1)
            anchors_list.append(anchors.reshape(-1, self.box_ndim))
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(
                np.full([num_anchors], match_thresh, anchors.dtype))
            unmatch_list.append(
                np.full([num_anchors], unmatch_thresh, anchors.dtype))
            idx += 1
        anchors = np.concatenate(anchors_list, axis=0)
        matched_thresholds = np.concatenate(match_list, axis=0)
        unmatched_thresholds = np.concatenate(unmatch_list, axis=0)
        return {
            "anchors": anchors,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds
        }

    def generate_anchors_dict(self, feature_map_size):
        ndim = len(feature_map_size)
        anchors_list = []
        matched_thresholds = [
            a.match_threshold for a in self._anchor_generators
        ]
        unmatched_thresholds = [
            a.unmatch_threshold for a in self._anchor_generators
        ]
        match_list, unmatch_list = [], []
        anchors_dict = OrderedDict()
        for a in self._anchor_generators:
            anchors_dict[a.class_name] = {}
        if self._feature_map_sizes is not None:
            feature_map_sizes = self._feature_map_sizes
        else:
            feature_map_sizes = [feature_map_size] * len(self._anchor_generators)
        idx = 0
        for anchor_generator, match_thresh, unmatch_thresh, fsize in zip(
                self._anchor_generators, matched_thresholds,
                unmatched_thresholds, feature_map_sizes):
            if len(fsize) == 0:
                fsize = feature_map_size
                self._feature_map_sizes[idx] = feature_map_size

            anchors = anchor_generator.generate(fsize)
            anchors = anchors.reshape([*fsize, -1, self.box_ndim])
            anchors = anchors.transpose(ndim, *range(0, ndim), ndim + 1)
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(
                np.full([num_anchors], match_thresh, anchors.dtype))
            unmatch_list.append(
                np.full([num_anchors], unmatch_thresh, anchors.dtype))
            class_name = anchor_generator.class_name
            anchors_dict[class_name]["anchors"] = anchors.reshape(-1, self.box_ndim)
            anchors_dict[class_name]["matched_thresholds"] = match_list[-1]
            anchors_dict[class_name]["unmatched_thresholds"] = unmatch_list[-1]
            idx += 1
        return anchors_dict

    @property
    def num_anchors_per_location(self):
        num = 0
        for a_generator in self._anchor_generators:
            num += a_generator.num_anchors_per_localization
        return num

    @property
    def box_ndim(self):
        return self._anchor_generators[0].ndim

    def num_anchors(self, class_name):
        if isinstance(class_name, int):
            class_name = self._classes[class_name]
        assert class_name in self._classes
        class_idx = self._classes.index(class_name)
        ag = self._anchor_generators[class_idx]
        feature_map_size = self._feature_map_sizes[class_idx]
        return np.prod(feature_map_size) * ag.num_anchors_per_localization

    def anchors_range(self, class_name):
        if isinstance(class_name, int):
            class_name = self._classes[class_name]
        assert class_name in self._classes
        num_anchors = 0
        anchor_ranges = []
        for name in self._classes:
            anchor_ranges.append((num_anchors, num_anchors + self.num_anchors(name)))
            num_anchors += anchor_ranges[-1][1] - num_anchors
        return anchor_ranges[self._classes.index(class_name)]
        
    def num_anchors_per_location_class(self, class_name):
        if isinstance(class_name, int):
            class_name = self._classes[class_name]
        assert class_name in self._classes
        class_idx = self._classes.index(class_name)
        return self._anchor_generators[class_idx].num_anchors_per_localization

