import numpy as np
from BiSAM_CD import compute_mask_iou_batch


def sum_masks_dict(masks_A, masks_B=None, iou_threshold=0.5):
    """
    直接对两mask求和（值可能超过1或255）,并执行归一化
    返回:
        sum_mask: 相同shape的矩阵，值为 mask1 + mask2
    """
    # 处理空输入
    if not masks_A and (masks_B is None or not masks_B):
        # 获取参考shape（若无法获取，抛出异常或指定默认shape）
        try:
            ref_shape = next(iter(masks_A.values())).shape
        except StopIteration:
            ref_shape = (1, 1024, 1024)  # 默认shape
        return np.zeros(ref_shape, dtype=np.uint8)

    try:
        merged_mask = np.zeros_like(next(iter(masks_A.values())), dtype=np.uint8)
    except StopIteration:
        ref_shape = (1, 1024, 1024)  # 默认shape
        merged_mask = np.zeros(ref_shape, dtype=np.uint8)

    # 没有对比的masks，直接返回合并后的mask
    if masks_B is None:
        for mask in masks_A.values():
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)
        return merged_mask

    # 将 masks_A 和 masks_B 转换为 NumPy 数组
    mask_array_A = np.array([m > 0 for m in masks_A.values()])
    mask_array_B = np.array([m > 0 for m in masks_B.values()])

    # 计算所有 mask 对的 IoU
    iou_matrix = compute_mask_iou_batch(mask_array_A, mask_array_B)

    # 找出需要删除的 key
    keys_to_remove = {"A": [], "B": []}
    for idx_A, obj_id_A in enumerate(masks_A.keys()):
        for idx_B, obj_id_B in enumerate(masks_B.keys()):
            if iou_matrix[idx_A, idx_B] > iou_threshold:
                if obj_id_A not in keys_to_remove["A"]:
                    keys_to_remove["A"].append(obj_id_A)
                if obj_id_B not in keys_to_remove["B"]:
                    keys_to_remove["B"].append(obj_id_B)

    for obj_id, mask in masks_A.items():
        if obj_id not in keys_to_remove["A"]:
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)

    for obj_id, mask in masks_B.items():
        if obj_id not in keys_to_remove["B"]:
            merged_mask = np.logical_or(merged_mask, mask > 0).astype(np.uint8)

    return merged_mask
