from second.core.target_assigner import TargetAssigner,LabelAssigner


def build(model_cfg, bv_range, box_coder):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if 'NoAnchor' not in model_cfg.rpn.module_class_name:
        target_assigner = TargetAssigner(
            model_cfg,
            box_coder=box_coder,
            )
    else:
        target_assigner = LabelAssigner(
            model_cfg = model_cfg,
            box_coder=box_coder
            )
    return target_assigner
