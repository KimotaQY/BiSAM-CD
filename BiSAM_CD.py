import gc
import json
import os
from pathlib import Path
import shutil

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from utils.extract_masks import extract_masks
from pycocotools import mask as mask_utils


def linear_color_interpolation(img1, img2, alpha):
    """
    线性颜色插值
    :param img1: T1图像（BGR格式）
    :param img2: T2图像（BGR格式）
    :param alpha: 插值权重（0为全T1，1为全T2）
    :return: 中间帧图像
    """
    # 提取RGB通道（忽略Alpha）
    img1_rgb = img1[:, :, :3]
    img2_rgb = img2[:, :, :3]

    # 线性插值
    interpolated_rgb = (1 - alpha) * img1_rgb + alpha * img2_rgb
    interpolated_rgb = interpolated_rgb.astype(np.uint8)
    return interpolated_rgb


def gen_frame(folder_paths, output_dir="output_jpg", sort="asc", mid_frame=0):
    # 根据排序方式决定遍历顺序
    paths_to_process = folder_paths if sort == "asc" else list(reversed(folder_paths))

    # 清空文件夹内容
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # 确保输出文件夹存在（只需要检查一次）
    os.makedirs(output_dir, exist_ok=True)

    for idx, folder_path in enumerate(paths_to_process):
        # 构造输入和输出路径
        input_path = folder_path
        output_filename = f"{idx + 1}.jpg" if idx == 0 else f"{idx + mid_frame + 1}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # 打开PNG图片并转换为RGB模式（JPG不支持PNG的RGBA透明度）
        filename = os.path.basename(folder_path)
        try:
            with Image.open(input_path) as img:
                if img.mode in ("RGBA", "LA"):
                    # 创建一个白色背景的RGB图像
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # 保存为JPG
                img.save(output_path, "JPEG", quality=100)
                print(f"转换成功: {filename} -> {os.path.basename(output_path)}")
        except Exception as e:
            print(f"转换失败 {filename}: {str(e)}")

    def generate_uniform_alphas(num_frames):
        """生成均匀间隔的alpha值"""
        return [i / (num_frames + 1) for i in range(1, num_frames + 1)]

    # 生成中间帧
    alphas = generate_uniform_alphas(mid_frame)
    for idx, alpha in enumerate(alphas):
        # 构造输入和输出路径
        input_path = paths_to_process[0]
        output_filename = f"{idx + 2}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # 打开PNG图片并转换为RGB模式（JPG不支持PNG的RGBA透明度）
        try:
            with Image.open(input_path) as img:
                if img.mode in ("RGBA", "LA"):
                    # 创建一个白色背景的RGB图像
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                first_frame = paths_to_process[0]
                final_frame = paths_to_process[-1]
                img = linear_color_interpolation(
                    cv2.imread(first_frame, cv2.IMREAD_UNCHANGED),
                    cv2.imread(final_frame, cv2.IMREAD_UNCHANGED),
                    alpha=alpha,
                )
                # 保存为JPG
                cv2.imwrite(output_path, img)
                # img.save(output_path, "JPEG", quality=100)
                print(f"中间帧生成: {alpha} -> {os.path.basename(output_path)}")
        except Exception as e:
            print(f"中间帧生成失败 {alpha}: {str(e)}")

    return output_dir


def add_new_obj(
    ann_frame_idx,
    ann_obj_id,
    points=None,
    labels=None,
    box=None,
    mask=None,
    predictor=None,
    inference_state=None,
):
    try:
        ann_frame_idx = ann_frame_idx  # the frame index we interact with
        ann_obj_id = ann_obj_id  # give a unique id to each object we interact with (it can be any integers)

        if points is not None or box is not None:
            # Let's add a positive click at (x, y) to get started
            points = np.array(points, dtype=np.float32) if points is not None else None
            # for labels, `1` means positive click and `0` means negative click
            labels = np.array(labels, np.int32) if labels is not None else None

            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
                box=box,
            )

        if mask is not None:
            # 1. 将 OpenCV 掩码 (0,255) 转换为二进制 (0,1)
            binary_mask = (mask > 128).astype(np.uint8)  # 阈值化

            # 2. 转换为 PyTorch 张量，并转为布尔类型
            mask_tensor = torch.from_numpy(binary_mask).to(torch.bool)

            # 检查形状是否为 (H, W)
            assert mask_tensor.dim() == 2, f"Mask must be 2D, got {mask_tensor.shape}"

            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                mask=mask_tensor,
            )
    except Exception as e:
        raise e  # 主动抛出错误

    return _, out_obj_ids, out_mask_logits


def compute_mask_iou(mask1, mask2):
    """
    计算两mask的IoU（交并比）差异
    返回:
        iou: 相似度（0~1，1表示完全相同）
        diff_mask: 差异区域（1表示不同，0表示相同）
    """
    intersection = np.logical_and(mask1 > 0, mask2 > 0)
    union = np.logical_or(mask1 > 0, mask2 > 0)
    sum_union = np.sum(union)
    if sum_union == 0:  # 两个 mask 都是全 0，认为完全相同
        return 1.0
    iou = np.sum(intersection) / sum_union
    # diff_mask = np.logical_xor(mask1 > 0, mask2 > 0).astype(np.uint8)
    return iou


def compute_mask_iou_batch(masks1, masks2):
    """
    计算两组 mask 的 IoU 矩阵 (num_masks1 x num_masks2)
    masks1: shape (num_masks1, H, W)
    masks2: shape (num_masks2, H, W)
    return: IoU matrix of shape (num_masks1, num_masks2)
    """
    # 边界条件处理：检查是否为空（适用于 NumPy arrays 和 lists）
    if isinstance(masks1, np.ndarray) and masks1.size == 0:
        return np.zeros((0, len(masks2)))
    if isinstance(masks2, np.ndarray) and masks2.size == 0:
        return np.zeros((len(masks1), 0))
    if not isinstance(masks1, np.ndarray) and not masks1:
        return np.zeros((0, len(masks2)))
    if not isinstance(masks2, np.ndarray) and not masks2:
        return np.zeros((len(masks1), 0))

    # Flatten masks to binary vectors
    masks1 = masks1.astype(bool).reshape(len(masks1), -1)  # (N1, H*W)
    masks2 = masks2.astype(bool).reshape(len(masks2), -1)  # (N2, H*W)

    # Compute intersection and union
    intersection = masks1 @ masks2.T  # (N1, N2)
    union = (
        np.sum(masks1, axis=1)[:, None] + np.sum(masks2, axis=1)[None, :] - intersection
    )

    # Avoid division by zero
    iou = intersection / union
    return iou


def merge_masks(masks_dict, compare_masks_dict=None, iou_threshold=0.5):
    """
    合并当前帧的masks，但跳过与对比帧中高IoU的物体

    参数:
        masks_dict (dict): 当前帧的masks {obj_id: mask}
        compare_masks_dict (dict): 对比帧的masks {obj_id: mask}（可选）
        iou_threshold (float): IoU阈值，大于此值则跳过合并

    返回:
        merged_mask (dict): 保留下来的mask
    """
    merged_mask = {}

    # 如果没有对比帧，直接返回masks_dict
    if compare_masks_dict is None:
        return masks_dict

    # 遍历当前帧的每个物体
    for obj_id, mask in masks_dict.items():
        mask_binary = (mask > 0).astype(np.uint8)

        # 检查对比帧中是否存在高IoU的物体
        compare_mask = compare_masks_dict.get(obj_id)
        compare_binary = (compare_mask > 0).astype(np.uint8)

        # 计算IoU（忽略全零mask的情况）
        if np.any(compare_binary) or np.any(mask_binary):
            iou = compute_mask_iou(compare_binary.flatten(), mask_binary.flatten())
            # if iou < 1.0:
            #     print(f"iou: {iou}")
            if iou <= iou_threshold:
                # 仅合并低IoU的物体
                # print("合并")
                merged_mask[obj_id] = mask
                # 显示每个obj的iou
                # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                # show_mask(mask, ax1, obj_id=obj_id)
                # ax1.set_title(f"IoU {iou} (Masks)")
                # plt.tight_layout()
                # plt.show()

    return merged_mask


def step_one(
    img_paths: list,
    annos_list: list,
    predictor=None,
    mid_frame=0,
    diff_frame_num=1,
    iou_threshold=0.5,
    prompt_type="box",
    prompts={},
):
    #
    diff_mask_list = []

    for i, annos in enumerate(annos_list):
        # 生成顺序jpg
        video_dir = gen_frame(
            img_paths,
            sort="asc" if i == 0 else "desc",
            mid_frame=mid_frame,
        )

        # scan all the JPEG frame names in this directory
        frame_names = [
            p
            for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=video_dir)

        # track objects
        predictor.reset_state(inference_state)

        print(f"建筑物数量: {len(annos)}")

        # 每次单独追踪一个对象
        def single_building_predict():
            # 获取追踪结果
            video_segments = (
                {}
            )  # video_segments contains the per-frame segmentation results
            for idx, item in enumerate(annos):
                mask = item.get("mask")
                box = item.get("box")
                points = item.get("points")

                ann_list = []
                for frame_idx in range(mid_frame + 1):
                    if prompt_type == "points":
                        # 使用points
                        labels = [1]  # ***标记正负样本点***
                        ann_list.append(
                            {
                                "ann_frame_idx": frame_idx,
                                "ann_obj_id": idx + 1,
                                "points": points,
                                "labels": labels,
                            }
                        )
                    elif prompt_type == "box":
                        # 使用box
                        ann_list.append(
                            {
                                "ann_frame_idx": frame_idx,
                                "ann_obj_id": idx + 1,
                                "box": box,
                            }
                        )
                    else:
                        # 使用mask
                        ann_list.append(
                            {
                                "ann_frame_idx": frame_idx,
                                "ann_obj_id": idx + 1,
                                "mask": mask,
                            }
                        )

                # 每栋建筑单独预测
                # 将ann_list导入predictor
                try:
                    for index, item in enumerate(ann_list):
                        _, out_obj_ids, out_mask_logits = add_new_obj(
                            **item, predictor=predictor, inference_state=inference_state
                        )

                except Exception as e:
                    raise e

                if len(ann_list) != 0:
                    for (
                        out_frame_idx,
                        out_obj_ids,
                        out_mask_logits,
                    ) in predictor.propagate_in_video(inference_state):
                        if out_frame_idx not in video_segments:
                            video_segments[out_frame_idx] = {}
                        for i, out_obj_id in enumerate(out_obj_ids):
                            video_segments[out_frame_idx][out_obj_id] = (
                                (out_mask_logits[i] > 0.0).cpu().numpy()
                            )

                predictor.reset_state(inference_state)

            return video_segments

        # 一次追踪seg_len个对象
        def predict_buildings_by_seglen(seg_len=50):
            video_segments = (
                {}
            )  # video_segments contains the per-frame segmentation results

            for frame_idx in range(mid_frame + 1):

                # 将masks按seg_len进行分段
                segment = []
                for i in range(0, len(annos), seg_len):
                    segment.append(annos[i : i + seg_len])
                    # print(segment)

                if len(annos) > 100:
                    print("too many objects")

                for seg_idx, seg_masks in enumerate(segment):
                    ann_list = []
                    for idx, item in enumerate(seg_masks):
                        mask = item.get("mask")
                        box = item.get("box")
                        points = item.get("points")

                        if prompt_type == "points":
                            # 使用points
                            labels = [1]  # ***标记正负样本点***
                            ann_list.append(
                                {
                                    "ann_frame_idx": frame_idx,
                                    "ann_obj_id": idx + 1 + seg_idx * seg_len,
                                    "points": points,
                                    "labels": labels,
                                }
                            )
                        elif prompt_type == "box":
                            # 使用box
                            ann_list.append(
                                {
                                    "ann_frame_idx": frame_idx,
                                    "ann_obj_id": idx + 1 + seg_idx * seg_len,
                                    "box": box,
                                }
                            )
                        else:
                            # 使用mask
                            # binary_mask = mask_utils.decode(rle)
                            ann_list.append(
                                {
                                    "ann_frame_idx": frame_idx,
                                    "ann_obj_id": idx + 1 + seg_idx * seg_len,
                                    "mask": mask,
                                }
                            )

                    # 将ann_list导入predictor
                    try:
                        for index, item in enumerate(ann_list):
                            _, out_obj_ids, out_mask_logits = add_new_obj(
                                **item,
                                predictor=predictor,
                                inference_state=inference_state,
                            )
                    except Exception as e:
                        raise e

                    if len(ann_list) != 0:
                        for (
                            out_frame_idx,
                            out_obj_ids,
                            out_mask_logits,
                        ) in predictor.propagate_in_video(inference_state):
                            if out_frame_idx not in video_segments:
                                video_segments[out_frame_idx] = {}
                            for i, out_obj_id in enumerate(out_obj_ids):
                                video_segments[out_frame_idx][out_obj_id] = (
                                    (out_mask_logits[i] > 0.0).cpu().numpy()
                                )

                    predictor.reset_state(inference_state)

            return video_segments

        video_segments = predict_buildings_by_seglen()
        # mask合并显示
        segments_len = len(video_segments)
        if segments_len == 0:
            diff_mask = {}
        else:
            # 首尾帧比较
            diff_mask = merge_masks(
                video_segments[0 if diff_frame_num == 1 else segments_len - 2],
                compare_masks_dict=video_segments[segments_len - 1],
                iou_threshold=iou_threshold,
            )

        diff_mask_list.append(diff_mask)

        # 消融实验用
        # diff_mask_list.append(video_segments)

        torch.cuda.empty_cache()  # 清理 PyTorch 的 CUDA 缓存

    # 显式释放 predictor
    del predictor
    gc.collect()

    return diff_mask_list
