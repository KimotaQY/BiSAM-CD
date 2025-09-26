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
    Linear color interpolation between two images

    :param img1: T1 image (BGR format)
    :param img2: T2 image (BGR format)
    :param alpha: Interpolation weight (0 for full T1, 1 for full T2)
    :return: Interpolated frame image
    """
    # Extract RGB channels (ignoring Alpha)
    img1_rgb = img1[:, :, :3]
    img2_rgb = img2[:, :, :3]

    # Linear interpolation
    interpolated_rgb = (1 - alpha) * img1_rgb + alpha * img2_rgb
    interpolated_rgb = interpolated_rgb.astype(np.uint8)
    return interpolated_rgb


def gen_frame(folder_paths, output_dir="output_jpg", sort="asc", mid_frame=0):
    """
    Convert PNG format image files to JPEG format and optionally generate intermediate interpolated frames

    This function iterates through the input folder path list, converts PNG images to JPEG format,
    and generates intermediate interpolated frames as needed. The main processing includes
    image format conversion (RGBA/LA to RGB), file renaming, and color interpolation.

    Parameters:
        folder_paths (list): List of paths to PNG image files
        output_dir (str): Directory path for output JPEG images, defaults to "output_jpg"
        sort (str): File processing order, "asc" for ascending order, other values for descending, defaults to "asc"
        mid_frame (int): Number of intermediate frames to generate, defaults to 0 (no intermediate frames)

    Returns:
        str: Output directory path
    """
    # Determine traversal order based on sorting method
    paths_to_process = folder_paths if sort == "asc" else list(reversed(folder_paths))

    # Clear folder contents
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Ensure output folder exists (only need to check once)
    os.makedirs(output_dir, exist_ok=True)

    # Process all input image files
    for idx, folder_path in enumerate(paths_to_process):
        # Construct input and output paths
        input_path = folder_path
        output_filename = f"{idx + 1}.jpg" if idx == 0 else f"{idx + mid_frame + 1}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Open PNG image and convert to RGB mode (JPEG does not support PNG's RGBA transparency)
        filename = os.path.basename(folder_path)
        try:
            with Image.open(input_path) as img:
                if img.mode in ("RGBA", "LA"):
                    # Create an RGB image with white background
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(
                        img, mask=img.split()[-1]
                    )  # Use alpha channel as mask
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # Save as JPEG
                img.save(output_path, "JPEG", quality=100)
                print(
                    f"Conversion successful: {filename} -> {os.path.basename(output_path)}"
                )
        except Exception as e:
            print(f"Conversion failed {filename}: {str(e)}")

    def generate_uniform_alphas(num_frames):
        """Generate uniformly spaced alpha values"""
        return [i / (num_frames + 1) for i in range(1, num_frames + 1)]

    # Generate intermediate frames
    alphas = generate_uniform_alphas(mid_frame)
    for idx, alpha in enumerate(alphas):
        # Construct input and output paths
        input_path = paths_to_process[0]
        output_filename = f"{idx + 2}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Open PNG image and convert to RGB mode (JPEG does not support PNG's RGBA transparency)
        try:
            with Image.open(input_path) as img:
                if img.mode in ("RGBA", "LA"):
                    # Create an RGB image with white background
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(
                        img, mask=img.split()[-1]
                    )  # Use alpha channel as mask
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")

                # Perform linear color interpolation to generate intermediate frames
                first_frame = paths_to_process[0]
                final_frame = paths_to_process[-1]
                img = linear_color_interpolation(
                    cv2.imread(first_frame, cv2.IMREAD_UNCHANGED),
                    cv2.imread(final_frame, cv2.IMREAD_UNCHANGED),
                    alpha=alpha,
                )
                # Save as JPEG
                cv2.imwrite(output_path, img)
                # img.save(output_path, "JPEG", quality=100)
                print(
                    f"Intermediate frame generated: {alpha} -> {os.path.basename(output_path)}"
                )
        except Exception as e:
            print(f"Intermediate frame generation failed {alpha}: {str(e)}")

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
    """
    Add a new object for segmentation to the model

    This function supports specifying objects to be segmented via points, bounding boxes, or masks.
    Depending on the input type provided, it calls the appropriate model method to add the new object
    and generate segmentation results.

    Args:
        ann_frame_idx (int): The frame index to interact with
        ann_obj_id (int): A unique ID assigned to each interacting object (can be any integer)
        points (list, optional): List of point coordinates in format [[x1, y1], [x2, y2], ...]
        labels (list, optional): List of point labels, where 1 indicates a positive click and 0 indicates a negative click
        box (list, optional): Bounding box coordinates in format [x1, y1, x2, y2]
        mask (numpy.ndarray, optional): Binary mask array used to specify the object region
        predictor: Predictor object used to perform segmentation operations
        inference_state: Inference state object containing the state information required for model inference

    Returns:
        tuple: A tuple containing three elements:
            - _: Model prediction results (unnamed return value)
            - out_obj_ids (list): List of output object IDs
            - out_mask_logits (torch.Tensor): Logits tensor of the output masks

    Raises:
        Exception: Raises an exception if an error occurs during the process of adding a new object
    """
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
            # 1. Convert OpenCV mask (0,255) to binary (0,1)
            binary_mask = (mask > 128).astype(np.uint8)  # Thresholding

            # 2. Convert to PyTorch tensor and then to boolean type
            mask_tensor = torch.from_numpy(binary_mask).to(torch.bool)

            # Check if the shape is (H, W)
            assert mask_tensor.dim() == 2, f"Mask must be 2D, got {mask_tensor.shape}"

            _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                mask=mask_tensor,
            )
    except Exception as e:
        raise e  # Explicitly raise the error

    return _, out_obj_ids, out_mask_logits


def compute_mask_iou(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) between two masks

    This function measures the similarity between two binary masks by computing
    the ratio of their intersection area to their union area. The IoU value
    ranges from 0 to 1, where 1 indicates identical masks and 0 indicates
    no overlap.

    Args:
        mask1 (numpy.ndarray): First mask array where non-zero values
                              represent foreground regions
        mask2 (numpy.ndarray): Second mask array where non-zero values
                              represent foreground regions

    Returns:
        float: IoU value between the two masks in range [0, 1]
               Returns 1.0 when both masks are all zeros (considered identical)
    """
    intersection = np.logical_and(mask1 > 0, mask2 > 0)
    union = np.logical_or(mask1 > 0, mask2 > 0)
    sum_union = np.sum(union)
    if sum_union == 0:  # Both masks are all zeros, considered identical
        return 1.0
    iou = np.sum(intersection) / sum_union
    # diff_mask = np.logical_xor(mask1 > 0, mask2 > 0).astype(np.uint8)
    return iou


def compute_mask_iou_batch(masks1, masks2):
    """
    Compute the IoU matrix between two sets of masks.

    This function calculates the Intersection over Union (IoU) between each pair of masks
    from two batches of masks. IoU is a common metric for evaluating the similarity between
    masks, particularly in computer vision tasks.

    Args:
        masks1: First set of masks with shape (num_masks1, H, W) where num_masks1 is the
                number of masks in the first set and H, W are the height and width of each mask
        masks2: Second set of masks with shape (num_masks2, H, W) where num_masks2 is the
                number of masks in the second set and H, W are the height and width of each mask

    Returns:
        numpy.ndarray: IoU matrix of shape (num_masks1, num_masks2) where each element (i, j)
                      represents the IoU between the i-th mask in masks1 and j-th mask in masks2
    """
    # Handle edge cases for empty inputs
    if isinstance(masks1, np.ndarray) and masks1.size == 0:
        return np.zeros((0, len(masks2)))
    if isinstance(masks2, np.ndarray) and masks2.size == 0:
        return np.zeros((len(masks1), 0))
    if not isinstance(masks1, np.ndarray) and not masks1:
        return np.zeros((0, len(masks2)))
    if not isinstance(masks2, np.ndarray) and not masks2:
        return np.zeros((len(masks1), 0))

    # Flatten masks to binary vectors for efficient computation
    masks1 = masks1.astype(bool).reshape(len(masks1), -1)  # (N1, H*W)
    masks2 = masks2.astype(bool).reshape(len(masks2), -1)  # (N2, H*W)

    # Compute intersection using matrix multiplication
    intersection = masks1 @ masks2.T  # (N1, N2)

    # Compute union using inclusion-exclusion principle
    union = (
        np.sum(masks1, axis=1)[:, None] + np.sum(masks2, axis=1)[None, :] - intersection
    )

    # Calculate IoU avoiding division by zero
    iou = intersection / union
    return iou


def merge_masks(masks_dict, compare_masks_dict=None, iou_threshold=0.5):
    """
    Merge masks from current frame, skipping objects with high IoU in the comparison frame

    Parameters:
        masks_dict (dict): Masks from current frame {obj_id: mask}
        compare_masks_dict (dict): Masks from comparison frame {obj_id: mask} (optional)
        iou_threshold (float): IoU threshold, objects with IoU higher than this value will be skipped

    Returns:
        merged_mask (dict): Retained masks
    """
    merged_mask = {}

    # If there is no comparison frame, return masks_dict directly
    if compare_masks_dict is None:
        return masks_dict

    # Iterate through each object in the current frame
    for obj_id, mask in masks_dict.items():
        # Convert mask to binary image with non-zero elements as 1 and zero elements as 0
        mask_binary = (mask > 0).astype(np.uint8)

        # Check if there is an object with the same ID in the comparison frame
        compare_mask = compare_masks_dict.get(obj_id)
        # Also convert the mask in the comparison frame to binary image
        compare_binary = (compare_mask > 0).astype(np.uint8)

        # Calculate IoU (ignoring cases where masks are all zeros)
        if np.any(compare_binary) or np.any(mask_binary):
            # Calculate the IoU value between two masks
            iou = compute_mask_iou(compare_binary.flatten(), mask_binary.flatten())
            # If IoU is less than or equal to threshold, keep the mask
            if iou <= iou_threshold:
                # Only merge objects with low IoU
                merged_mask[obj_id] = mask

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

        print(f"object num: {len(annos)}")

        # Tracking one object at a time
        def single_building_predict():
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
                        labels = [1]  # sign the positive and negative samples
                        ann_list.append(
                            {
                                "ann_frame_idx": frame_idx,
                                "ann_obj_id": idx + 1,
                                "points": points,
                                "labels": labels,
                            }
                        )
                    elif prompt_type == "box":
                        ann_list.append(
                            {
                                "ann_frame_idx": frame_idx,
                                "ann_obj_id": idx + 1,
                                "box": box,
                            }
                        )
                    else:
                        ann_list.append(
                            {
                                "ann_frame_idx": frame_idx,
                                "ann_obj_id": idx + 1,
                                "mask": mask,
                            }
                        )

                # add all obj to predictor
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

        # Tracking seg_len objects at a time
        def predict_buildings_by_seglen(seg_len=50):
            video_segments = (
                {}
            )  # video_segments contains the per-frame segmentation results

            for frame_idx in range(mid_frame + 1):

                # Segment the masks according to seg_len
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
                            labels = [1]  # sign the positive and negative samples
                            ann_list.append(
                                {
                                    "ann_frame_idx": frame_idx,
                                    "ann_obj_id": idx + 1 + seg_idx * seg_len,
                                    "points": points,
                                    "labels": labels,
                                }
                            )
                        elif prompt_type == "box":
                            ann_list.append(
                                {
                                    "ann_frame_idx": frame_idx,
                                    "ann_obj_id": idx + 1 + seg_idx * seg_len,
                                    "box": box,
                                }
                            )
                        else:
                            ann_list.append(
                                {
                                    "ann_frame_idx": frame_idx,
                                    "ann_obj_id": idx + 1 + seg_idx * seg_len,
                                    "mask": mask,
                                }
                            )

                    # add all obj to predictor
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

        # merge masks
        segments_len = len(video_segments)
        if segments_len == 0:
            diff_mask = {}
        else:
            # compare the first and last frames to get the difference
            diff_mask = merge_masks(
                video_segments[0 if diff_frame_num == 1 else segments_len - 2],
                compare_masks_dict=video_segments[segments_len - 1],
                iou_threshold=iou_threshold,
            )

        diff_mask_list.append(diff_mask)

        # For ablation experiments
        # diff_mask_list.append(video_segments)

        torch.cuda.empty_cache()

    del predictor
    gc.collect()

    return diff_mask_list
